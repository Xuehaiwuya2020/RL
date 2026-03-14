import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig
)
from trl import PPOTrainer, PPOConfig
from safetensors.torch import load_file
import swanlab
import json
import os
from typing import List
import copy

# ==========================================
# 1. 配置与初始化
# ==========================================
swanlab.login(api_key="FD1UV1wgT0yOnkP30rdGd", save=True)
swanlab.config.update({
    "model": "Qwen_PPO/Qwen3-0.6B",
    })
swanlab.init()

class Config:
    # 模型路径
    model_name = "../output_sft/sft-model"  # SFT 模型路径
    reward_model_path = "../rm_models/rm_model" # 训练好的 RM 路径
    
    # 数据路径
    data_path = "data/ppo_train.json"
    # 输出路径
    output_dir = "../ppo_models"

    device = "cuda:0"
    
    # 训练参数
    learning_rate = 1.4e-6
    batch_size = 4 
    mini_batch_size = 2 
    gradient_accumulation_steps = 2
    
    # PPO 特定参数
    ppo_epochs = 2
    init_kl_coef = 0.2
    target_kl = 6.0 
    gamma = 1
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.1 
    
    # 生成参数
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": 151643, 
        "max_new_tokens": 64,   
    }

# ==========================================
# 2. 定义 MultiDimensionRewardModel (用于 Critic 和 Reward Model)
# ==========================================
class MultiDimensionRewardModel(nn.Module):
    def __init__(self, model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            output_hidden_states=True,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        hidden_size = self.model.config.hidden_size
        
        # 定义四个维度的打分头
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(hidden_size, 1),
            'relevance': nn.Linear(hidden_size, 1),
            'coherence': nn.Linear(hidden_size, 1),
            'quality': nn.Linear(hidden_size, 1)
        })
        
        # 将打分头转换为半精度并移动到设备
        self.score_heads.to(torch.bfloat16)
        self.score_heads.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取最后一个 token 的 hidden state
        last_token_hidden = last_hidden_state[:, -1, :]
        
        # 计算四个维度的分数
        scores = []
        for head_name, head in self.score_heads.items():
            scores.append(head(last_token_hidden))

        scores = torch.cat(scores, dim=-1)  # [batch_size, 4]
        
        # 这里我们返回聚合后的分数 (Value)，用于 PPO 的 Value Function
        # 通常取平均值或加权和
        value = scores.mean(dim=-1, keepdim=True) 
        
        return value

    def get_scores(self, input_ids, attention_mask):
        """获取原始分数，用于计算 Reward"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        
        scores = []
        for head in self.score_heads.values():
            scores.append(head(last_token_hidden))

        scores = torch.cat(scores, dim=-1)  # [batch_size, 4]
        return scores

    def load_reward_weights(self, save_directory):
        """加载训练好的 Reward Model 权重"""
        model_path = os.path.join(save_directory, "model.safetensors")
        if not os.path.exists(model_path):
            print(f"Warning: Reward model weights not found at {model_path}")
            return

        state_dict = load_file(model_path)
        
        # 加载 score_heads 的权重
        model_state_dict = self.state_dict()
        matched_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                matched_state_dict[k] = v
            elif f"score_heads.{k}" in model_state_dict:
                matched_state_dict[f"score_heads.{k}"] = v
                
        missing_keys, unexpected_keys = self.load_state_dict(matched_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading reward model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading reward model: {unexpected_keys}")
            
        print("Reward model weights loaded successfully.")

# ==========================================
# 3. 数据处理
# ==========================================
class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f: 
                if not line.strip():
                    continue
                raw_data.append(json.loads(line))
                       
        for item in raw_data:
            conversations = item["conversations"]
            history_turns = conversations[:-1]
            messages = []
            for turn in history_turns:
                messages.append({"role": turn["role"], "content": turn["content"]})
            
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            target_response = conversations[-1]["content"]
            
            self.data.append({
                "prompt": prompt,
                "response": target_response
            })
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    config = Config()
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备 Reward Model (用于打分)
    reward_model = MultiDimensionRewardModel(config.model_name, device=config.device)
    reward_model.load_reward_weights(config.reward_model_path)
    reward_model.eval() 
    # 完全冻结 Reward Model
    for param in reward_model.parameters():
        param.requires_grad = False

    # 3. 准备 Actor 模型 (Policy)
    # 使用 AutoModelForCausalLM，不带 Value Head
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    actor_model.train()

    # 4. 准备 Critic 模型 (Value)
    # 结构与 Reward Model 完全一致
    critic_model = MultiDimensionRewardModel(config.model_name, device=config.device)
    critic_model.load_reward_weights(config.reward_model_path)
    
    # --- 关键步骤：冻结 Critic 的 Base Model，只训练 Score Heads ---
    # 这样 Critic 就是一个基于 RM 初始化的、可微调的 Value Function
    for name, param in critic_model.named_parameters():
        if "score_heads" not in name:
            param.requires_grad = False
            
    print("Critic model trainable parameters:")
    for name, param in critic_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    # 5. 准备数据
    dataset = PPODataset(config.data_path, tokenizer)
    
    def collator(data):  
        return {key: [d[key] for d in data] for key in data[0]}
    
    # 6. 初始化 PPO 配置
    ppo_config = PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        init_kl_coef=config.init_kl_coef,
        gamma=config.gamma,
        lam=config.lam,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        vf_coef=config.vf_coef
    )
    
    # 7. 初始化 PPO Trainer
    # 注意：这里我们传入自定义的 critic_model
    # PPOTrainer 会调用 critic_model 的 forward 方法来获取 Value
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        tokenizer=tokenizer, 
        model=actor_model,
        ref_model=None, 
        critic_model=critic_model, # 传入我们的 MultiDimensionRewardModel 实例
        dataset=dataset,
        data_collator=collator
    )
    
    # 8. 生成奖励函数
    def compute_rewards(texts: List[str]) -> torch.Tensor:
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(reward_model.device)
        
        with torch.no_grad():
            # 获取四个维度的分数 [batch, 4]
            scores = reward_model.get_scores(inputs['input_ids'], inputs['attention_mask'])
            
            # 计算总分：简单平均 (也可以根据需要调整权重)
            total_score = scores.mean(dim=-1)
            
        return total_score

    # 9. 训练循环
    generation_kwargs = config.gen_kwargs.copy()
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    print("开始训练...")
    
    global_step = 0
    
    for epoch in range(config.ppo_epochs):
        for step, batch in enumerate(ppo_trainer.dataloader):
            # 获取 query tensors
            query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"].squeeze(0).to(ppo_trainer.accelerator.device) for q in batch["prompt"]]
            
            # 生成回复
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # 解码
            batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            batch["query"] = batch["prompt"] 
            
            # 计算奖励
            texts_to_score = [q + r for q, r in zip(batch["query"], batch["response"])]
            rewards = compute_rewards(texts_to_score)
            
            # 转换为 CPU 列表
            rewards = [r.cpu() for r in rewards]
            
            # 运行 PPO 步骤
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # 记录日志
            ppo_trainer.log_stats(stats, batch, rewards)
            
            # 记录到 SwanLab
            swanlab.log({
                "reward/mean": sum(rewards)/len(rewards),
                "reward/dist": rewards,
                "ppo/policy/ratio": stats["ppo/policy/ratio"],
                "ppo/loss/total": stats["ppo/loss/total"],
                "ppo/loss/policy": stats["ppo/loss/policy"],
                "ppo/loss/value": stats["ppo/loss/value"],
            }, step=global_step)
            
            print(f"Epoch {epoch}, Step {step}: Mean Reward = {sum(rewards)/len(rewards):.4f}")
            
            global_step += 1
            
            # 保存模型
            if global_step % 100 == 0:
                print(f"Saving checkpoint at step {global_step}")
                # 保存 Actor 模型
                actor_model.save_pretrained(os.path.join(config.output_dir, f"actor_checkpoint_step_{global_step}"))
                # 保存 Critic 模型的 score_heads (Base Model 是冻结的，不需要保存)
                torch.save(critic_model.score_heads.state_dict(), os.path.join(config.output_dir, f"critic_score_heads_step_{global_step}.pth"))
                tokenizer.save_pretrained(os.path.join(config.output_dir, f"checkpoint_step_{global_step}"))

    # 10. 保存最终模型
    print("训练完成，保存模型...")
    actor_model.save_pretrained(os.path.join(config.output_dir, "final_actor_model"))
    torch.save(critic_model.score_heads.state_dict(), os.path.join(config.output_dir, "final_critic_score_heads.pth"))
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
