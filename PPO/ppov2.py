import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl.experimental.ppo import PPOTrainer, PPOConfig
from safetensors.torch import load_file
import swanlab
import json
import os
from copy import deepcopy

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
    learning_rate = 1.4e-7
    batch_size = 4  # batch_size = mini_batch_size * gradient_accumulation_steps
    mini_batch_size = 2 
    gradient_accumulation_steps = 2
    
    # PPO 特定参数
    ppo_epochs = 3
    init_kl_coef = 0.1
    target_kl = 0.2 
    gamma = 1
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.2 
    
    # 生成参数
    gen_kwargs = {
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 64,   
    }

# ==========================================
# 2. 统一的打分模型 (同时适配 Reward 和 Value)
# ==========================================
class MultiDimensionScoreModel(nn.Module):
    """
    统一模型：
    - is_value_model=False -> 作为 Reward Model，输出 [batch_size, 1]
    - is_value_model=True  -> 作为 Value Model (Critic)，输出 [batch_size, seq_len, 1]
    并包装为 SequenceClassifierOutput 完美欺骗 trl 的接口。
    """
    base_model_prefix = "model"
    
    def __init__(self, model_path, device='cuda:0', is_value_model=False):
        super().__init__()
        self.device = device
        self.is_value_model = is_value_model
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16, 
            output_hidden_states=True,
            device_map="auto",
            use_safetensors=True
        )
        
        # 继承 config 防止 trl 底层报错
        self.config = self.model.config
        hidden_size = self.config.hidden_size
        
        # 定义四个维度的打分头
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(hidden_size, 1),
            'relevance': nn.Linear(hidden_size, 1),
            'coherence': nn.Linear(hidden_size, 1),
            'quality': nn.Linear(hidden_size, 1)
        })
        
        self.score_heads.to(torch.bfloat16)
        self.score_heads.to(self.device)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 获取所有 Token 的隐状态: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1] 
        
        if self.is_value_model:
            # 【作为 Critic】计算每个 Token 的价值
            scores = []
            for head in self.score_heads.values():
                scores.append(head(hidden_states)) # 每个 head: [batch, seq_len, 1]
            
            scores = torch.cat(scores, dim=-1) # [batch, seq_len, 4]
            value = scores.mean(dim=-1, keepdim=True) # 聚合为 [batch, seq_len, 1]
            
            return SequenceClassifierOutput(logits=value)
            
        else:
            # 【作为 Reward Model】只计算最后一个 Token 的最终奖励
            # (基于 left-padding，最后一个 token 就是最右边的 token)
            last_token_hidden = hidden_states[:, -1, :] # [batch_size, hidden_size]
            
            scores = []
            for head in self.score_heads.values():
                scores.append(head(last_token_hidden))
                
            scores = torch.cat(scores, dim=-1)  # [batch, 4]
            value = scores.mean(dim=-1, keepdim=True) # 聚合为 [batch, 1]
            
            return SequenceClassifierOutput(logits=value)


def load_reward_weights(base_model_path, reward_checkpoint_path, model_class, device="cuda:0", is_value_model=False):
    print(f"Initializing {'Value' if is_value_model else 'Reward'} model from base model: {base_model_path}")
    model = model_class(base_model_path, device=device, is_value_model=is_value_model)
    
    model_path = os.path.join(reward_checkpoint_path, "model.safetensors")
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = load_file(model_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    else:
        print(f"Warning: {model_path} not found. Using randomly initialized weights.")

    return model

# ==========================================
# 3. 数据处理 (保持不变，适应新版 PPO 仅需 input_ids)
# ==========================================
class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        raw_data = []
        self.tokenizer.padding_side = "left"
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f: 
                if not line.strip(): continue
                raw_data.append(json.loads(line))
                        
        for item in raw_data:
            conversations = item["conversations"]
            history_turns = conversations[:-1]
            messages = [{"role": turn["role"], "content": turn["content"]} for turn in history_turns]
            
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=False,
                truncation=True, 
                max_length=512 
            )
            
            self.data.append({
                "input_ids": inputs["input_ids"].squeeze(0),
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
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 准备 Actor 模型 (Policy) - 直接用官方标准的 CausalLM!
    print("Loading Actor Model...")
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager" 
    )
    actor_model.generation_config.pad_token_id = tokenizer.pad_token_id
    actor_model.generation_config.eos_token_id = tokenizer.eos_token_id
    actor_model.train()

    # 3. 准备 Ref 模型 (冻结的 Actor 副本)
    print("Preparing Reference Model...")
    ref_model = deepcopy(actor_model)
    ref_model.eval()

    # 4. 准备 Reward Model (提供最终标量奖励，is_value_model=False)
    print("Loading Reward Model...")
    reward_model = load_reward_weights(config.model_name, config.reward_model_path, MultiDimensionScoreModel, config.device, is_value_model=False)
    reward_model.eval() 

    # 5. 准备 Critic 模型 (提供 Token 级价值，is_value_model=True)
    print("Loading Critic Model...")
    critic_model = load_reward_weights(config.model_name, config.reward_model_path, MultiDimensionScoreModel, config.device, is_value_model=True)
    
    # 冻结 Critic 的基座，仅训练 4 个 score_heads 
    for name, param in critic_model.named_parameters():
        if "score_heads" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # Value head 需要随训练更新
    critic_model.train()

    # 6. 准备数据
    dataset = PPODataset(config.data_path, tokenizer)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) 

    # 7. 初始化 PPO 配置
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_ppo_epochs=config.ppo_epochs,
        kl_coef=config.init_kl_coef,
        target_kl=config.target_kl,
        gamma=config.gamma,
        lam=config.lam,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        vf_coef=config.vf_coef,
        # 传递你的生成参数
        **config.gen_kwargs 
    )
    
    # 8. 初始化 PPO Trainer
    print("Initializing PPO Trainer...")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer, 
        model=actor_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=critic_model,  # 传入我们的 Token 级别 Value 模型
        train_dataset=dataset,
        data_collator=collator
    )
    
    # 9. 一键训练
    print("开始训练...")
    ppo_trainer.train()

    # 10. 保存最终模型
    print("训练完成，保存模型...")
    actor_model.save_pretrained(os.path.join(config.output_dir, "final_actor_model"))
    
    # 因为 Critic 模型被更新了，所以保存它最新的 score_heads 状态
    torch.save(critic_model.score_heads.state_dict(), os.path.join(config.output_dir, "final_value_heads.pth"))
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()