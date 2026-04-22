# Description: This file contains utility functions for the InternVL2 model.
# Partially taken from: https://huggingface.co/OpenGVLab/InternVL2-1B

import importlib.util
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import to_absolute_path
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig

from huggingface_hub import snapshot_download

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_num_image_tokens_per_patch(encoder_variant: str) -> int:
    # we want to know how many image tokens we use so that we can adjust the batch padding
    # 读取模型配置 作用：下载并读取HuggingFace模型的config.json
    tmp_config = AutoConfig.from_pretrained(encoder_variant, trust_remote_code=True)  #
    # 确定输入图像尺寸
    image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
    # 读取 patch size (每个块的大小)
    patch_size = tmp_config.vision_config.patch_size
    # 计算 token 数量
    num_image_tokens = int((image_size // patch_size) ** 2 * (tmp_config.downsample_ratio ** 2))
    return num_image_tokens

def get_assistant_loss_mask(user_starts, assistant_starts, prompt_tokenized_ids):
    # assistant_start_end = []
    seq_length = prompt_tokenized_ids.shape[1]
    loss_mask = torch.zeros(prompt_tokenized_ids.shape, dtype=torch.bool) # loss is calculated where this mask is True
    
    for batch_id, (user_list, assistant_list) in enumerate(zip(user_starts, assistant_starts)):
        # batch_pairs = []
        # assume we start with user always:
        assert user_list[0] < assistant_list[0], "First user start should be before first assistant start"
        assert len(user_list) == len(assistant_list), "Number of user and assistant starts should be the same"
        
        for i, start in enumerate(assistant_list):
            # End is the start of the next user sequence OR the last index if it's the final sequence
            end = user_list[i + 1] - 1 if i < len(user_list) - 1 else seq_length - 1  # updated variable name to seq_length
            # batch_pairs.append((start, end))
            loss_mask[batch_id, start:end+1] = True

        # assistant_start_end.append(batch_pairs)
    return loss_mask


def get_chat_tokens(tokenizer, prompts: List[str], user_start_token_str: str, assistant_start_token_str: str) -> Dict:


    """
    prompts = ["<|im_start|>user\n<img><IMG_CONTEXT>...省略...</img>\nCurrent speed: 9.8 m/s. Command: follow the road. Predict the waypoints.<|im_end|><|im_start|>assistant\nWaypoints:<|im_end|>"]

    user_start_token_str = "<|im_start|>user"
    assistant_start_token_str = "<|im_start|>assistant"

    将对话 prompt 转换为 token id，并构造基于角色的 loss mask,使模型仅在 assistant 响应上进行监督学习
    """
    
    
    
    # 一、将整条字符串(prompts)变为token id
    prompt_tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)  # 利用tokenizer把 prompt 转换成 prompt_tokenized，并进行 padding 以适应 batch 计算
    prompt_tokenized_ids = prompt_tokenized["input_ids"]   # 取出 prompt_tokenized 的 token id, 形状为[B,L]
    # prompts = ["<|im_start|>user\n<img><IMG_CONTEXT>...<IMG_CONTEXT></img>\nCurrent speed: 9.8 m/s. Command: follow the road. Predict the waypoints.<|im_end|><|im_start|>assistant\nWaypoints:<|im_end|>"] 
    # 这里的 token id 是根据 tokenizer 的词表把 prompt 转换成数字表示，具体的数字会根据 tokenizer 的词表而不同。
    # prompt_tokenized_ids: 
    # tensor([[151644,    872,    198, 151646, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
    #          151648, 151648, 151648, 151647,    198,   5405,   4628,     25,    220,
    #              16,     15,     13,     20,    296,   2687,     13,   7348,     25,
    #            1795,    279,   5636,     13,  32710,    279,  81286,     13, 151645,
    #          151644,  77091,    198]])    
    
    
    
    
    
    
    # 二、 因为上面我们进行了padding,所以这里来构造prompt_tokenized_valid和prompt_tokenized_mask,以标记哪些位置是有效的token，哪些位置是padding token
    # 这里讲一下为什么需要pad,因为在训练时需要把一个 batch 内的多个样本放在一起进行计算,而这些样本的长度可能不一样,所以需要把它们 pad 到同样的长度,这样才能形成一个矩阵进行批量计算.
    # 但是对于batch size=1的情况,实际上是没有真正pad的,因为只有一个样本,不需要与其他样本对齐
    # 判断每个token是否不是padding token,得到一个布尔矩阵，形状为[B,L]，其中 True 表示这个位置的 token 不是 padding，False 表示是 padding
    # 对于 batch size=1 的情况，这个矩阵中所有位置都是 True，因为没有真正的 padding   
    # prompt_tokenized_valid = [True, True, True, True, ..., True]
    prompt_tokenized_valid = prompt_tokenized["input_ids"] != tokenizer.pad_token_id
    prompt_tokenized_mask = prompt_tokenized_valid

    # mask user prompt (question) to calculate loss only on assistant tokens (answer)





    ##########################  真正关键的从这里开始  ##########################
    



    # 三、编码 user / assistant 起始 token
    # user_start_token_str = "<|im_start|>user" 是用户消息的起始标记，assistant_start_token_str = "<|im_start|>assistant" 是助手消息的起始标记
    user_start_token_ids = torch.tensor(tokenizer(user_start_token_str)["input_ids"])
    assistant_start_token_ids = torch.tensor(tokenizer(assistant_start_token_str)["input_ids"])
    # print(f"user_start_token_ids: {user_start_token_ids}, assistant_start_token_ids: {assistant_start_token_ids}")
    # user_start_token_ids: tensor([151644,872,198]), assistant_start_token_ids: tensor([151644,77091,198])

    
    
    
    
    # 四、获取user_start_token_ids和assistant_start_token_ids的长度  
    # 因为我们要在 prompt_tokenized_ids 上滑动一个窗口来匹配 user_start_token_ids 和 assistant_start_token_ids，所以需要知道这两个 token 序列的长度，才能确定窗口的大小。
    seq_len_to_find = user_start_token_ids.shape[0]                 # 3
    seq_len_to_find_assistant = assistant_start_token_ids.shape[0]  # 3
    
    
    
    
    
    # 五、在整条序列里“扫描” user 起点和 assistant 起点
    # 同样的操作也应用于 assistant_start_token_ids
    matches_user = (prompt_tokenized_ids.unfold(1, seq_len_to_find, 1) == user_start_token_ids).all(dim=2)                     
    matches_assistant = (prompt_tokenized_ids.unfold(1, seq_len_to_find_assistant, 1) == assistant_start_token_ids).all(dim=2)  
    # print(f"matches_user: {matches_user}, matches_assistant: {matches_assistant}")
    # matches_user      = [[ True, False, False, False, ..., False ]]  表示在第i个位置出现了user token
    # matches_assistant = [[ False, False, False, False, ..., True ]]  表示在第j个位置出现了assistant token
    
    
    
    
    
    
    
    
    
    # 六、获取匹配索引
    match_indices_user = torch.nonzero(matches_user, as_tuple=True)             
    match_indices_assistant = torch.nonzero(matches_assistant, as_tuple=True)
    # match_indices_user: (tensor([0]), tensor([0])), match_indices_assistant: (tensor([0]), tensor([540]))
    # match_indices_user 的第一个 tensor 表示匹配到的 batch id，第二个 tensor 表示匹配到的位置索引.
    # 因为我们只有一个样本，所以 batch id 都是 0，位置索引分别是 0（user 起点）和 540（assistant 起点）

    # tuple(batch id), tuple(start index) -> dict key: batch id, value: list of start indices
    position_user_start_indices = [[] for _ in range(len(match_indices_user[0]))]
    position_assistant_start_indices = [[] for _ in range(len(match_indices_assistant[0]))]
    for i in range(len(match_indices_user[0])):
        batch_id = match_indices_user[0][i].item()
        position_user_start_indices[batch_id].append(match_indices_user[1][i].item())
    for i in range(len(match_indices_assistant[0])):
        batch_id = match_indices_assistant[0][i].item()
        position_assistant_start_indices[batch_id].append(match_indices_assistant[1][i].item())
    # position_user_start_indices: [[0]], position_assistant_start_indices: [[540]]
    # 为什么要写成“列表的列表”？因为作者这个函数设计成支持多轮对话:user1 -> assistant1 -> user2 -> assistant2 -> ...  那样一个样本里就可能有多个 user 起点和多个 assistant 起点
    # 虽然 SimLingo 当前通常只用单轮问答，但这个函数保留了多轮能力

    
    
    
    
    
    # 七、最后一步：根据这些位置(user的起点位置->position_user_start_indices, assistant的起点位置->position_assistant_start_indices)索引来构造 loss mask
    # 构造这个loss mask 是为了只让 assistant 的回答部分计算 loss
    loss_mask = get_assistant_loss_mask(position_user_start_indices, position_assistant_start_indices, prompt_tokenized_ids)
    # loss_mask: tensor([[False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #          False, False, False, False, False, False, False, False, False, False,
    #           True,  True,  True]])


    return {
        'phrase_ids': prompt_tokenized_ids,       # token id 形状为[B,L]
        'phrase_valid': prompt_tokenized_valid,   # 标记哪些token不是padding 形状为[B,L]
        'phrase_mask': prompt_tokenized_mask,     # 直接把有效token的位置作为mask 形状为[B,L]
        'language_string': prompts,               # 原始文本 形状为[B]
        'loss_masking': loss_mask                 # 哪些位置计算loss
    }


def get_custom_chat_template(conversations: List[Dict], tokenizer, encoder_variant: str, num_image_tokens_total: int, cache_root_dir: str = 'pretrained') -> Optional[Dict]:
    # get the custom chat template
    # for full conversation, question only
    # https://huggingface.co/docs/transformers/main/en/chat_templating#can-i-use-chat-templates-in-training
    # this adds special tokens and bring it in the right format for the pretrained LLM
        
    # taken from:
    # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/train/constants.py#L7
    # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L294
    # IMG_START_TOKEN='<img>'
    # IMG_END_TOKEN='</img>'
    # IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    # IMG_TOKEN = '<image>'

    # cache_dir = f"{cache_root_dir}/{(encoder_variant.split('/')[1])}"
    # # get absolute path from workspace dir not wokring dir
    # cache_dir = to_absolute_path(cache_dir)
    # model_path = f"{cache_dir}/conversation.py"
    # if not os.path.exists(model_path):
    #     from huggingface_hub import snapshot_download
    #     # snapshot_download(repo_id=encoder_variant, local_dir=cache_dir)
    #     if os.path.isdir(encoder_variant):
    #         model_path = encoder_variant
    #     else:
    #         model_path = snapshot_download(repo_id=encoder_variant, local_dir=cache_dir)
        
    # #import from file from model_path
    # spec = importlib.util.spec_from_file_location('get_conv_template', model_path)
    # conv_module = importlib.util.module_from_spec(spec)
    # sys.modules['get_conv_template'] = conv_module
    # spec.loader.exec_module(conv_module)


    """

    conversations = 
    [
    [
     {'role': 'user', 
      'content': [
                  {'type': 'text', 
                   'text': 'Current speed: 10.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.'
                  }, 
                  {'type': 'image'
                  }
                 ]
     }, 

     {'role': 'assistant', 
      'content': [
                  {'type': 'text', 
                   'text': 'Waypoints:'
                  }
                 ]
     }
    ]
    ]

    cache_root_dir:
        本地缓存模型的根目录，默认是 pretrained,这里会在这个目录下创建一个子目录来存放下载的模型，例如 pretrained/InternVL2-1B
        如果 encoder_variant 不是本地路径，而是 HuggingFace repo 名称,这里会把模型下载到cache_root_dir缓存.
    """


    # 图像相关特殊token
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_TOKEN = '<image>'




    
    # 一、找到 InternVL2 模型目录，目的是为了从中读取 conversation.py 文件
    #     情况1: encoder_variant 是一个本地目录，直接使用
    if os.path.isdir(encoder_variant):  #  encoder_variant=/root/InternVL2-1B
        model_dir = encoder_variant
    #     情况2: encoder_variant 是 HuggingFace 模型标识，从 HuggingFace Hub 下载到本地缓存目录
    else:
        cache_dir = f"{cache_root_dir}/{encoder_variant.split('/')[1]}"              # 构造缓存路径,例如 pretrained/InternVL2-1B
        cache_dir = to_absolute_path(cache_dir)                                      # 转换为绝对路径，确保在任何工作目录下都能正确访问
        model_dir = snapshot_download(repo_id=encoder_variant, local_dir=cache_dir)  # 从 HuggingFace 下载整个模型仓库到这个目录
    # 二、定位并动态导入 conversation.py
    #     不自己硬编码 chat template，而是直接调用 InternVL2 官方模型仓库里的 conversation.py
    # 这样做的好处是：
    #     模板格式和原模型保持一致
    #     不容易写错特殊分隔符
    #     更适配预训练模型原始对话风格
    model_path = os.path.join(model_dir, "conversation.py")  # model_path = /root/InternVL2-1B/conversation.py  找到模型自带的对话模板文件
    if not os.path.exists(model_path):                       # 如果没有找到,直接报错
        raise FileNotFoundError(f"conversation.py not found: {model_path}")
    spec = importlib.util.spec_from_file_location('get_conv_template', model_path) # 根据文件路径创建模块规格
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {model_path}")
    conv_module = importlib.util.module_from_spec(spec)  # 创建一个 Python 模块对象
    sys.modules['get_conv_template'] = conv_module       # 真正执行 conversation.py，加载里面的函数和类
    spec.loader.exec_module(conv_module)                 # 调用 InternVL2 官方 conversation.py 里定义的模板函数

    
    
    
    
    
    
    
    
    # 构造图像占位模板串 例如 <img><IMG_CONTEXT><IMG_CONTEXT>...</img> 其中 IMG_CONTEXT 根据 num_image_tokens_total 决定数量
    image_tokens_templates = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens_total + IMG_END_TOKEN







    # 一、准备两个 prompt 列表  这两个列表分别保存 batch 中每个样本构造出的两种 prompt
    prompts_conv = []     # 1. 完整对话模板（包含问题和答案）这个主要用于训练
    prompts_question = [] # 2. 只有问题的模板（答案部分是空的）这个主要用于推理，评估时也可以用来计算损失mask
    
    
    
    
    
    
    
    # 二、把原始 conversation 转换为模板内部的 message 结构
    # 开始按样本逐个处理,其中 conv 是某一个样本的对话，是长度为 2 的列表，包含两轮对话：第一轮是用户提问，第二轮是助手回答
    for idx, conv in enumerate(conversations):  
        
        # 强制限制:只支持两轮对话,这句说明作者这里实际上只打算处理(单轮问答：user + assistant),虽然 get_chat_tokens 那边写了支持多轮对话的逻辑，但这个函数这里直接限制成 2 轮
        assert len(conv) == 2, "For question and answer templates only two turn conversation (user + assistant) is supported. During training is should work but is not checked!!"
        
        
        
        
        # 创建两个相同的模板对象
        template_conv = conv_module.get_conv_template('internlm2-chat')       # 用于完整对话
        template_question = conv_module.get_conv_template('internlm2-chat')   # 用于仅问题模板

       
       
       
       
        # 三、向完整对话模板里追加消息
        # {"role": "user",      "content": [{"text": "..."}] } 这是每次取出来的 conv 的格式(问题)
        # {"role": "assistant", "content": [{"text": "..."}] } 这是每次取出来的 conv 的格式(答案)
        for conv_part_idx, conv_part in enumerate(conv):
            
            
            
            
            content_str = conv_part['content'][0]['text'] # 取出当前轮次原始prompt的文本内容   ！！！！！！！！！！！这是核心！！！！！！！！！！！！！
            
            
            
            
            # 取出来之后，根据角色(user/assistant)把文本内容追加到对应的模板里
            if conv_part['role'] == 'assistant':          # 如果当前轮是 assistant
                template_conv.append_message(template_conv.roles[1], content_str)  # ！！！把内容追加到模板中 assistant 的角色位置
            elif conv_part['role'] == 'user':                             # 如果当前轮是 user
                if conv_part_idx == 0 and IMG_TOKEN not in content_str:   # 如果这是第 1 轮用户消息,且用户文本中没有显式写 <image>
                    content_str = f"{IMG_TOKEN}\n" + content_str          # 那就自动在最前面补一个<image>
                    # 为什么这样做？因为 InternVL2 的预训练就是在用户消息的开头放一个<image>占位符，来告诉模型这里有图像输入，所以在 fine-tuning 时也要保持这个格式，这样才能更好地利用预训练学到的知识
                template_conv.append_message(template_conv.roles[0], content_str)  # ！！！把内容追加到模板中 user 的角色位置
            else:   # 如果 role 不是 user 或 assistant，直接报错
                raise ValueError(f"Role {conv_part['role']} not supported")
        
        
        
        
        
        
        # 四、向问题模板里追加消息
        # 要求第一轮必须是用户提问,因为 question 模板就是从第一轮 user 消息构造出来的
        assert conv[0]['role'] == 'user', "First turn should be user as this should be the question."
        content_str_user = conv[0]['content'][0]['text']  # 取出第一轮用户消息的文本内容
        if IMG_TOKEN not in content_str_user:   # 如果没有 <image>，同样自动补上 说明不管训练还是推理，模型都希望 prompt 里出现图像占位
            content_str_user = f"{IMG_TOKEN}\n" + content_str_user
        template_question.append_message(template_question.roles[0], content_str_user)  # ！！！把内容追加到模板中 user 的角色位置
        template_question.append_message(template_question.roles[1], None)  # 模板里保留 assistant 开始回答的位置，但答案内容为空，留给模型去生成

        
        
        
        
        
        
        # 五、把模板对象变成真正的 prompt 字符串,这里会根据 conversation.py 里定义的格式，把消息拼成真正的字符串
        prompt_conv = template_conv.get_prompt()
        prompt_question = template_question.get_prompt()

        # prompt_conv 打印结果显示如下格式:
        # <|im_start|>system
        # 你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user
        # <image>
        # Current speed: 9.8 m/s. Command: follow the road. Predict the waypoints.<|im_end|><|im_start|>assistant
        # Waypoints:<|im_end|>



        # 六、把模板默认自带的 system prompt 删掉
        # replace system prompt to reduce tokens and save memory
        # template_conv.system_template -> '<|im_start|>system\n{system_message}'
        system_prompt = template_conv.system_template.replace('{system_message}', template_conv.system_message) + template_conv.sep
        prompt_conv = prompt_conv.replace(system_prompt, '')
        prompt_question = prompt_question.replace(system_prompt, '')

        # prompt_conv 打印结果显示如下格式:
        # <|im_start|>user
        # <image>
        # Current speed: 9.8 m/s. Command: follow the road. Predict the waypoints.<|im_end|><|im_start|>assistant
        # Waypoints:<|im_end|>

        # 七、把 <image> 替换成真正的视觉 token 模板
        # replace <image> with image token placeholders
        prompt_conv = prompt_conv.replace(IMG_TOKEN, image_tokens_templates, 1)
        prompt_question = prompt_question.replace(IMG_TOKEN, image_tokens_templates, 1)

        # <|im_start|>user
        # <img><IMG_CONTEXT><IMG_CONTEXT>...</img>
        # Current speed: 9.8 m/s. Command: follow the road. Predict the waypoints.<|im_end|><|im_start|>assistant
        # Waypoints:<|im_end|>


        # 八、保存当前样本的两个 prompt
        prompts_conv.append(prompt_conv)
        prompts_question.append(prompt_question)
        # 处理完当前样本后，把结果加入 batch 列表。
        # 所以循环结束后：
        # prompts_conv：batch 内所有样本的完整对话 prompt
        # prompts_question：batch 内所有样本的问题 prompt

    
    
    
    # on list of prompts to get the padding right
    user_start_token_str = template_conv.roles[0]
    assistant_start_token_str = template_conv.roles[1]
    # print(f"user_start_token_str: {user_start_token_str}, assistant_start_token_str: {assistant_start_token_str}")
    # user_start_token_str: <|im_start|>user, assistant_start_token_str: <|im_start|>assistant






    # 九、把 prompt 批量 token 化，并生成 mask
    conv_dict = get_chat_tokens(tokenizer, prompts_conv, user_start_token_str, assistant_start_token_str)
    question_dict = get_chat_tokens(tokenizer, prompts_question, user_start_token_str, assistant_start_token_str)



    """
        conv_dict = {
        'phrase_ids': prompt_tokenized_ids,       # token id 形状为[B,L]
        'phrase_valid': prompt_tokenized_valid,   # 标记哪些token不是padding 形状为[B,L]
        'phrase_mask': prompt_tokenized_mask,     # 直接把有效token的位置作为mask 形状为[B,L]
        'language_string': prompts,               # 原始文本 形状为[B]  包括问题和答案！！！！！！！！！！！！
        'loss_masking': loss_mask                 # 哪些位置计算loss
    }

        question_dict = {
        'phrase_ids': prompt_tokenized_ids,       # token id 形状为[B,L]
        'phrase_valid': prompt_tokenized_valid,   # 标记哪些token不是padding 形状为[B,L]
        'phrase_mask': prompt_tokenized_mask,     # 直接把有效token的位置作为mask 形状为[B,L]
        'language_string': prompts,               # 原始文本 形状为[B]  包括问题！！！！！！！！！！！！
        'loss_masking': loss_mask                 # 哪些位置计算loss
    }
    """

    return conv_dict, question_dict



def preprocess_image_batch(
        images_batch_list, 
        input_size=448, 
        use_global_img=False, 
        max_num_grid=2
    ):
    """
    把一个 batch 的原始图像列表，转换成视觉编码器可以直接吃的标准化张量

    # images batch list 0: tensor([[[ 5., 20.,  7.,  ...,  0., 16.,  4.],
                            #  [ 7.,  6.,  0.,  ...,  4., 15., 19.],
                            #  [ 4.,  8.,  9.,  ...,  3., 10., 25.],
                            #  ...,
                            #  [72., 60., 54.,  ..., 22., 12.,  0.],
                            #  [52., 63., 50.,  ...,  7., 19.,  8.],
                            #  [42., 33., 15.,  ...,  6.,  0.,  5.]],

                            # [[ 2., 17., 19.,  ...,  0., 15.,  0.],
                            #  [ 0.,  5.,  0.,  ...,  0.,  8., 13.],
                            #  [ 8.,  8.,  2.,  ...,  3.,  0., 23.],
                            #  ...,
                            #  [40., 50., 43.,  ..., 19.,  6.,  0.],
                            #  [40., 42., 26.,  ...,  0.,  5.,  0.],
                            #  [29., 19., 17.,  ...,  3.,  9., 10.]],

                            # [[ 3., 10.,  9.,  ...,  5.,  6.,  6.],
                            #  [ 3.,  3.,  5.,  ...,  0.,  0.,  0.],
                            #  [ 2.,  0.,  0.,  ...,  0.,  0.,  7.],
                            #  ...,
                            #  [25., 27., 17.,  ...,  7.,  1.,  0.],
                            #  [35., 29.,  6.,  ...,  0.,  0.,  0.],
                            #  [21., 17.,  8.,  ...,  4.,  3.,  0.]]])
    """
    
    
    # 这是对图像进行预处理的函数
    # 1.将图像变为RGB格式 -> 2.将每个patch resize成固定尺寸(448x448) -> 3.由PIL Image格式(H,W,C)转成 PyTorch tensor格式(C,H,W) -> 4.按 ImageNet 均值方差归一化
    transform = build_transform(input_size=input_size)  # input_size=448 这里定义了每个 patch 最终要被 resize 成的尺寸，通常是 448x448




    images_processed_tmp = []   # 每张原图处理后的 patch 张量
    images_sizes_tmp = []       # 每张原图切分后 patch 的尺寸信息




    # 开始处理 batch 中的每张图像
    for idx, img in enumerate(images_batch_list):
        image_np = img.numpy().astype(np.uint8)        # 把 PyTorch tensor 变成 NumPy 数组，并强制转成 uint8
        image_np = np.transpose(image_np, (1, 2, 0))   # 调整维度顺序，从 (C, H, W) 变成 (H, W, C)，因为 PIL 处理的图像是 (H, W, C) 格式
        image = Image.fromarray(image_np)              # 变成 PIL 图像
        
        # 📸 把一张大图 → 切成很多小照片(这里是2张)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_global_img, max_num=max_num_grid)
        pixel_values = [transform(image) for image in images]  # 对每个 patch 做标准变换
        pixel_values = torch.stack(pixel_values)               # 把 patch 堆叠起来, 如果一张图有 2 个 patch，那么这里得到：[2, 3, 448, 448]
        images_processed_tmp.append(pixel_values)              # 保存这一张原图的处理结果,images_processed_tmp 里每个元素都代表一张原图处理后的 patch 张量
        images_sizes_tmp.append([image.size[1], image.size[0]])# 记录 patch 尺寸信息  最后一个 patch 的尺寸 [H, W]
    
    images_processed = {
        'pixel_values': torch.stack(images_processed_tmp), # 如果 batch 中总共有 N=BS*T 张图，每张图被切成 2 个 patch，那么：pixel_values.shape = [N, 2, 3, 448, 448]
        'image_sizes': torch.tensor(images_sizes_tmp)      # 形状为 [N, 2]，每行是一个 patch 的尺寸 [H, W]
        }
    return images_processed


def build_transform(input_size):
    """
    每个 patch 如何从 PIL Image 变成视觉模型输入张量
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD  # ImageNet 均值、方差，通常是 [0.485, 0.456, 0.406] 和 [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),       # 保证是 RGB 图像
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),  # 这里就是把每个 patch resize 成 448x448
        T.ToTensor(),                        # 转成 tensor, 把 PIL Image 转成 PyTorch tensor，格式从：[H, W, C]变成：[C, H, W]
        T.Normalize(mean=MEAN, std=STD)      # 按 ImageNet 均值方差归一化
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):

    """
    将任意尺寸、任意长宽比的图像,切分成若干个固定大小(448x448)的patch,供视觉模型输入使用

    返回：
        [
        patch1 (448×448),
        patch2 (448×448),
        ...
        patchN (448×448),
        (optional) thumbnail
        ]
    """





    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values