import os
from safetensors.torch import load_file

def load_reward_model(save_directory, pre_trained_path, model_class):
    # 1. 佈~[建模佞~K孾^佾K﻾H使潔¨顾D训纾C路彾D﻾I
    model = model_class(pre_trained_path)

    # 2. 佊| 载 safetensors 彝~C轇~M
    model_path = os.path.join(save_directory, "model.safetensors")

    state_dict = load_file(model_path)

    # 3. 佊| 载彝~C轇~M佈°模佞~K
    model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    pre_trained_path = 'output_sft/sft-model'
    save_path = 'rm_models/rm_model'
    model = MultiDimensionRewardModel(pre_trained_path, device='cuda:0')
    state_dict = load_file(os.path.join(save_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    '''
    model = load_reward_model( 
        pre_trained_path,
        save_path,
        MultiDimensionRewardModel
    )'''
    for name, param in model.score_heads.named_parameters():
      print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")