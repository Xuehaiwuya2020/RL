import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import swanlab
import json
import os
from typing import List
# ==========================================
# 1. 配置与初始化
# ==========================================
swanlab.login(api_key="FD1UV1wgT0yOnkP30rdGd", save=True)
swanlab.config.update({
    "model": "Qwen_PPO/Qwen3-0.6B",
    })

swanlab.init()  # swanlab 初始化
# 配置参数
class Config:
    # 模型路径 (请根据实际情况修改)
    model_name = "../output_sft/sft-model"  ## 或者你的本地路径 "D:\\model\\qwen\\qwen-0.6b"
    reward_model_path = "../rm_models/rm_model" # 训练好的RM路径
    
    # 数据路径
    data_path = "data/ppo_train.json" # 存放对话数据的JSON文件
    # 输出路径
    output_dir = "../ppo_models"

    device = "cuda:0"
    # 训练参数
    learning_rate = 1.4e-6
    batch_size = 4 # PPO batch size，Qwen模型较大，建议设小
    mini_batch_size = 2 # 梯度更新时的batch size
    gradient_accumulation_steps = 2
    
    # PPO 特定参数
    ppo_epochs = 2
    init_kl_coef = 0.2
    target_kl = 6.0 # 如果计算出的新旧策略之间KL散度超过这个值，说明策略更新步子过大 \
                # 会导致训练稳定，此时算法会停止当前的epoch更新(early stopping)或者增加KL系数
    gamma = 1
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2  # 防止价值函数更新步子过大，导致训练不稳定
    vf_coef = 0.1
    
    # 生成参数
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": 151643, # Qwen的pad token
        "max_new_tokens": 64,   # 生成回复的最大长度
    }

    # LoRA配置 (用于微调策略模型，减少显存占用)
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05


# 2. 定义 Reward Model (复用之前的定义)
class MultiDimensionRewardModel(nn.Module):
    def __init__(self, model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16, # 使用半精度节省显存
            output_hidden_states=True,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.hidden_size = self.model.config.hidden_size
        
        # 定义四个维度的打分头
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(self.hidden_size, 1),
            'relevance': nn.Linear(self.hidden_size, 1),
            'coherence': nn.Linear(self.hidden_size, 1),
            'quality': nn.Linear(self.hidden_size, 1)
        })
        
        self.score_heads.to(torch.float16)
        self.score_heads.to(self.device)
        # 加载训练好的打分头权重 (如果有)
        
        # 如果只保存了head，需要手动加载 state_dict

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取最后一个 token 的 hidden state
        # 注意：Qwen的padding通常在右侧，直接取 -1
        last_token_hidden = last_hidden_state[:, -1, :]
        
        # 计算四个维度的分数
        scores = []
        for head in self.score_heads.values():
            scores.append(head(last_token_hidden))

        scores = torch.cat(scores, dim=-1)  # [batch_size, 4]
        reward = scores.mean(dim=-1)  # 简单平均作为总奖励
        
        return reward


# 3. 数据处理
class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f: # 跳过空行
                if not line.strip():
                    continue
                raw_data.append(json.loads(line))
                       
        for item in raw_data:
            conversations = item["conversations"]
            # 提取历史对话作为 context
            # 我们的目标是训练模型回答最后一个 user 的问题
            history_turns = conversations[:-1]
            messages = []
            # 构造messages格式
            for turn in history_turns:
                messages.append({"role": turn["role"], "content": turn["content"]})
            
            # 使用tokenizer的 apply_chat_template 来格式化 prompt
            
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 将 query 拼接到 prompt 后面
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
def main():
    config = Config()
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # 2. 加载策略模型 (Actor) - 
    # 为了节省显存，我们使用 LoRA
    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        generation_config=AutoModelForCausalLM.from_pretrained(
        config.model_name
    ).generation_config
        # device_map="auto",
        # dtype=torch.float16,
        # trust_remote_code=True
    )
    
    # 配置 LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 针对 Qwen 的常见配置
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 将 LoRA 应用到基础模型
    # 注意：AutoModelForCausalLMWithValueHead 会自动包装模型
    # 我们需要先包装，再应用 LoRA，或者使用特定的方法
    # 这里我们使用 trl 推荐的方式：直接加载带 Value Head 的模型，然后应用 LoRA
    
    # critic 函数形式

    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path = config.model_name,
        # device_map="auto",
        # dtype=torch.float16,
        # trust_remote_code=True,
    )
    
    # model.base_model_prefix = "pretrained_model"
    # base_model.generation_config = AutoModelForCausalLM.from_pretrained(
    value_model.generation_config = value_model.pretrained_model.generation_config

    # 应用LoRA到预训练模型
    value_model.pretrained_model = get_peft_model(value_model.pretrained_model, peft_config)
    value_model.pretrained_model.print_trainable_parameters()
    
    # 3. 加载 Reward Model
    reward_model = MultiDimensionRewardModel(config.reward_model_path, device='cuda')
    reward_model.eval() # 奖励模型不需要训练
    
    # 4. 准备数据
    dataset = PPODataset(config.data_path, tokenizer)
    
    def collator(data):  # data: List[Dict[str, List[str]]]
        return {key: [d[key] for d in data] for key in data[0]}
    
    # 5. 初始化 PPO 配置
    ppo_config = PPOConfig(
        # model_name=config.model_name,
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
    
    #print("model类型是："+str(type(model)))
    # 6. 初始化 PPO Trainer
    from copy import deepcopy
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        tokenizer=tokenizer, # 处理文本输入的函数，这里直接使用 tokenizer
        model=base_model,
        ref_model=deepcopy(base_model), # 显存不够时，PPOTrainer会自动处理
        dataset=dataset,
        data_collator=collator
        # reward_model=reward_model,
        # value_model = reward_model,
        # peft_config=peft_config
    )
    
    # 7. 生成奖励函数
    # 输入: query (prompt) + response (generated text)
    # 输出: 标量奖励
    def compute_rewards(texts: List[str]) -> torch.Tensor:
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(reward_model.device)
        
        # 将输入移动到GPU
        inputs.to(config.device)
        with torch.no_grad():
            # 获取四个维度的分数
            total_score = reward_model(inputs['input_ids'], inputs['attention_mask'])
            
            # 计算总分
            # 策略1: 简单求和
            # total_score = scores_dict['consistency'] + scores_dict['relevance'] + scores_dict['coherence'] + scores_dict['quality']
            
            # 策略2: 加权平均 (假设权重相同)
            # total_score = (scores_dict['consistency'] + scores_dict['relevance'] + scores_dict['coherence'] + scores_dict['quality']) / 4.0
            
            # 策略3: 只使用 quality 分数
            # total_score = scores_dict['quality']
            
        return total_score

    # 8. 训练循环
    generation_kwargs = config.gen_kwargs.copy()
    # 确保 pad_token_id 正确
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    print("开始训练...")
    
    # 记录全局步数
    global_step = 0
    
    for epoch in range(config.ppo_epochs):
        for step, batch in enumerate(ppo_trainer.dataloader):
            # 获取 query tensors
            query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"].squeeze(0).to(ppo_trainer.accelerator.device) for q in batch["prompt"]]
            
            # 生成回复
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # 解码
            batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            batch["query"] = batch["prompt"] # prompt 已经是格式化过的
            
            # 计算奖励
            # 拼接 prompt 和 response 传给 Reward Model
            texts_to_score = [q + r for q, r in zip(batch["query"], batch["response"])]
            rewards = compute_rewards(texts_to_score)
            
            # 转换为 CPU 列表
            rewards = [r.cpu() for r in rewards]
            
            # 运行 PPO 步骤
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            # print(stats)
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
                # "ppo/loss/entropy": stats["ppo/loss/entropy"]
                # "kl": stats["objective/kl"]
            }, step=global_step)
            
            print(f"Epoch {epoch}, Step {step}: Mean Reward = {sum(rewards)/len(rewards):.4f}")
            
            global_step += 1
            
            # 保存模型
            if global_step % 100 == 0:
                print(f"Saving checkpoint at step {global_step}")
                # 保存 LoRA 权重
                base_model.pretrained_model.save_pretrained(os.path.join(config.output_dir, f"checkpoint_step_{global_step}"))
                tokenizer.save_pretrained(os.path.join(config.output_dir, f"checkpoint_step_{global_step}"))

    # 9. 保存最终模型
    print("训练完成，保存模型...")
    base_model.pretrained_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()