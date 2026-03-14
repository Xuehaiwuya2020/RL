import json
import random
from qwen_api import generate_with_qwen  # 确保你的 API 调用函数可用

def generate_negative_samples(conversation):
    """
    针对给定的对话片段生成 Prompt
    """
    if conversation[-1]['role'] != 'assistant':
        return None
    
    context_list = conversation[:-1]
    target_response = conversation[-1]['content']
    
    context_str = ""
    for turn in context_list:
        context_str += f"{turn['role'].capitalize()}: {turn['content']}\n"

    prompt_template = """
    你是一个极其严苛的多轮对话质量审核专家。
    我将给你一段对话上下文（Context）和一个完美的正样本回答（Chosen）。 
    请你基于这个正样本，伪造出 4 个具有特定缺陷的负样本，用于训练奖励模型。

    ### Context:
    {context}

    ### Chosen (正样本):
    {target_response}

    ### 任务：请生成以下四个负样本：
    1. Rejected_Consistency (一致性错误): 回复必须与前文提到的事实矛盾（如改错出生地、记错星座或身高）。
    2. Rejected_Relevance (相关性错误): 回复必须完全脱离当前用户的问题，聊无关话题。
    3. Rejected_Coherence (连贯性错误): 回复必须包含逻辑断层、语序颠倒或严重的病句。
    4. Rejected_Quality (回答质量错误): 回复必须态度敷衍、过于简短（如“哦”）或包含复读机行为。

    输出格式要求为严格的 JSON: 
    {{ "consistency": "...", "relevance": "...", "coherence": "...", "quality": "..." }}
    """
    return prompt_template.format(context=context_str, target_response=target_response)

def parse_json_res(res_str):
    """
    防止 DeepSeek 加上 ```json 等废话导致解析失败
    """
    try:
        # 尝试直接解析
        return json.loads(res_str)
    except:
        # 提取 JSON 块
        try:
            start = res_str.find('{')
            end = res_str.rfind('}') + 1
            return json.loads(res_str[start:end])
        except Exception as e:
            print(f"JSON解析彻底失败: {e}")
            return None

def process_sliding_window_sampling(raw_conv, num_samples=3):
    """
    实现滑动窗口+稀疏采样+随机起始逻辑
    """
    total_len = len(raw_conv)
    if total_len < 4: return [] # 对话太短不采

    # 确定采样锚点（早、中、晚）
    # 锚点是针对整个数组的索引位置
    anchors = [
        int(total_len * 0.3), 
        int(total_len * 0.6), 
        total_len - 1
    ]
    
    final_data = []
    
    for anchor in anchors:
        # 往回找最近的一个 assistant 回答
        idx = anchor
        while idx >= 0 and raw_conv[idx]['role'] != 'assistant':
            idx -= 1
        
        if idx < 2: continue # 至少保留两轮

        
        sub_conv = raw_conv[0:idx + 1]

        # 生成 Prompt 并调用
        prompt = generate_negative_samples(sub_conv)
        if not prompt: continue
        
        
        raw_res = generate_with_qwen(prompt)
        parsed_res = parse_json_res(raw_res)

        if parsed_res:
            # 构造最终存储格式
            sample = {
                "prompt": prompt.split("### Context:")[1].split("### Chosen")[0].strip(), # 提取纯净 context
                "chosen": sub_conv[-1]['content'],
                "rejected_consistency": parsed_res.get("consistency", ""),
                "rejected_relevance": parsed_res.get("relevance", ""),
                "rejected_coherence": parsed_res.get("coherence", ""),
                "rejected_quality": parsed_res.get("quality", "")
            }
            final_data.append(sample)
    
    return final_data


'''
raw_conv = [
    {"role": "user", "content": "你听说过比尔·奈伊这个人吗？"},
    {"role": "assistant", "content": "听说过啊，他是很有实力的一位演员。"},  
]
'''

def gen_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    with open("neg_train.json", "a", encoding="utf-8") as f_out:
        for i, item in enumerate(data):
            conv = item['conversations']
            results = process_sliding_window_sampling(conv) # 处理滑动窗口
        

            # 保存为 JSON
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                if i % 10 == 0:
                    print(f"\n已处理 {i} 条对话)")
                
   
    print(f"\n🎉 成功生成 {len(results)} 条多头负样本数据，已保存至 neg_train.json")

if __name__ == "__main__":
    data_path = "D:\\VsCodeProj\\RL\\RM\\data\\train.json"
    gen_data(data_path)