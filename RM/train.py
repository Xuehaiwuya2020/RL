import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)

import swanlab
import json

swanlab.login(api_key="FD1UV1wgT0yOnkP30rdGd", save=True)

swanlab.config.update({
    "model": "Qwen_RM/Qwen3-0.6B",
    })



# 1. 定义模型, 设置四个打分头
class MultiDimensionRewardModel(nn.Module):
    def __init__(self, model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            output_hidden_states=True,
            device_map=device
        )
        # 冻结基座模型参数（可选，如果显存不够或只想训练头）
        # for param in self.model.parameters():
        #     param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.hidden_size = self.model.config.hidden_size
        
        # 定义四个维度的打分头
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(self.hidden_size, 1),
            'relevance': nn.Linear(self.hidden_size, 1),
            'coherence': nn.Linear(self.hidden_size, 1),
            'quality': nn.Linear(self.hidden_size, 1)
        })

    def forward(self, input_ids, attention_mask, dimension='quality'):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取最后一个 token 的 hidden state
        # 注意：需要根据 attention_mask 找到真正的最后一个 token，防止 padding 干扰
        # 但为了简化且通常 padding 在后，这里直接取 -1
        last_token_hidden = last_hidden_state[:, -1, :]
        
        score = self.score_heads[dimension](last_token_hidden).squeeze(-1)
        return score

# ==========================================
# 2. 定义 Dataset
# ==========================================
class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
        raw_data = raw_data[:16]

        for item in raw_data:
            prompt = item['prompt']
            chosen = item['chosen']
            
            # 为每个维度创建一个训练样本
            dimensions = ['consistency', 'relevance', 'coherence', 'quality']
            for dim in dimensions:
                rejected_key = f'rejected_{dim}'
                
                self.data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': item[rejected_key],
                    'dimension': dim
                })
        
        print(f"Loaded {len(self.data)} training samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 拼接 prompt 和 response
        # 注意：这里假设模型不需要特殊的聊天模板拼接，如果是 ChatGLM/Qwen 等，需要按其格式拼接
        text_chosen = item['prompt'] + item['chosen']  # item chosen
        text_rejected = item['prompt'] + item['rejected']
        
        # Tokenize
        # 我们这里返回原始文本，在 collator 中进行 tokenize 和 padding
        # 这样可以动态适应 batch 中的最长序列
        return {
            'text_chosen': text_chosen,
            'text_rejected': text_rejected,
            'dimension': item['dimension']
        }

# ==========================================
# 3. 定义 Data Collator
# ==========================================
class RewardDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        dimensions = [item['dimension'] for item in batch]
        
        # 提取文本
        chosen_texts = [item['text_chosen'] for item in batch]
        rejected_texts = [item['text_rejected'] for item in batch]
        
        # Tokenize chosen
        chosen_inputs = self.tokenizer(
            chosen_texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Tokenize rejected
        rejected_inputs = self.tokenizer(
            rejected_texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            'input_ids_chosen': chosen_inputs['input_ids'],
            'attention_mask_chosen': chosen_inputs['attention_mask'],
            'input_ids_rejected': rejected_inputs['input_ids'],
            'attention_mask_rejected': rejected_inputs['attention_mask'],
            'dimension': dimensions
        }

# ==========================================
# 4. 定义 Trainer (重写 compute_loss)
# ==========================================
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 提取输入
        input_ids_chosen = inputs.get('input_ids_chosen')
        attention_mask_chosen = inputs.get('attention_mask_chosen')
        input_ids_rejected = inputs.get('input_ids_rejected')
        attention_mask_rejected = inputs.get('attention_mask_rejected')
        dimensions = inputs.get('dimension')
        
        # 前向传播梯度
        # 注意：由于一个 batch 中可能包含不同维度的样本，我们需要循环处理或分组处理
        # 为了简单起见，这里假设一个 batch 内的维度是一致的（或者我们逐个计算）
        # 更高效的做法是按维度分组，然后并行计算，但这里为了代码清晰使用循环
        
        batch_size = input_ids_chosen.size(0)
        loss_fct = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        
        # 将数据移到模型所在设备
        input_ids_chosen = input_ids_chosen.to(model.device)
        attention_mask_chosen = attention_mask_chosen.to(model.device)
        input_ids_rejected = input_ids_rejected.to(model.device)
        attention_mask_rejected = attention_mask_rejected.to(model.device)
        
        # 计算每个样本的损失
        # 注意：这里为了支持 batch 内混合维度，我们逐个计算
        # 如果显存紧张，建议在 Dataset 或 Collator 阶段就按维度分好组
        
        # 为了利用 GPU 并行，我们可以尝试向量化，但不同维度对应不同的 score_head
        # 这里采用折中方案：按维度分组计算
        
        # 收集所有维度的索引
        dim_indices = {}
        for i, dim in enumerate(dimensions):
            if dim not in dim_indices:
                dim_indices[dim] = []
            dim_indices[dim].append(i)
            '''
            {
            'consistency': [0, 3],  # 第0个和第3个样本属于一致性维度
            'relevance': [1],       # 第1个样本属于相关性维度
            'quality': [2]          # 第2个样本属于质量维度
            }
            '''
        
        for dim, indices in dim_indices.items():
            # 提取该维度的数据
            #  # indices 是列表，例如 [0, 3]，需要转为 tensor 才能用于索引
            idx_tensor = torch.tensor(indices).to(model.device)
            
            # 利用 index_select 从大 batch 中“抠”出属于当前维度的数据
            c_ids = input_ids_chosen.index_select(0, idx_tensor)
            c_mask = attention_mask_chosen.index_select(0, idx_tensor)
            r_ids = input_ids_rejected.index_select(0, idx_tensor)
            r_mask = attention_mask_rejected.index_select(0, idx_tensor)
            
            # 计算分数
            # forward 方法需要 dimension 参数
            chosen_scores = model(input_ids=c_ids, attention_mask=c_mask, dimension=dim)
            rejected_scores = model(input_ids=r_ids, attention_mask=r_mask, dimension=dim)
            
            # 计算损失: LogSigmoid(chosen - rejected)
            # 或者使用 BCE: Sigmoid(chosen - rejected) vs 1
            diff = chosen_scores - rejected_scores
            labels = torch.ones_like(diff) # 目标是 chosen > rejected
            loss = loss_fct(diff, labels)
            
            total_loss += loss * len(indices) # 加权平均
            
        avg_loss = total_loss / batch_size
        
        return (avg_loss, None) if return_outputs else avg_loss

# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    # 配置
    model_path = "D:\\model\\qwen\\qwen-0.6b" 
    data_path = "D:\\VsCodeProj\\RL\\RM\\data\\neg_train.json"
    output_dir = ".\\output_rm"
    
    # 初始化模型和 Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiDimensionRewardModel(model_path, device=device)
    tokenizer = model.tokenizer
    
    # 准备数据集
    train_dataset = PreferenceDataset(data_path, tokenizer)
    
    # 定义 TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # 根据 GPU 显存调整
        gradient_accumulation_steps=4,  # 模拟更大的 batch size
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False, # 重要！因为我们自定义了 collator
        report_to = "swanlab"  # 使用swanlab记录训练过程
    )
    
    # 初始化 Trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RewardDataCollator(tokenizer, max_length=1024)
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
