import torch
from torch import nn
from typing import List, Optional
from transformers import AutoModel

import torch.nn.functional as F

class LingoInternVLModel(nn.Module):
    
    
    def __init__(self, variant, *args, **kwargs):
        super().__init__()
        
        
        
        
        
        # 一、加载模型
        self.model = AutoModel.from_pretrained(variant, trust_remote_code=True)  # trust_remote_code=True 表示允许 HuggingFace 执行模型仓库里自定义的 Python 代码
        
        
        
        
        
        # 二、语言模型原始词表一共有多少 token embedding
        try:
            self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
        except:
            self.num_embeddings = self.model.language_model.vocab_size
        
        
        
        
        # 三、两个预留接口 
        self.use_global_img = None  # 预留接口
        self.processor = None       # 预留接口


        # 六视角视觉token压缩配置。
        # 每个相机有2个patch，每个patch的256个token排列为16×16。
        # 将每个patch池化为4×8，共32个token；
        # 因而每个相机保留64个token，六个相机共384个token。
        self.num_cameras = 6
        self.num_patches_per_camera = 2
        self.visual_pool_height = 4
        self.visual_pool_width = 8
        self.num_visual_tokens_per_patch = (
            self.visual_pool_height
            * self.visual_pool_width
        )
        
    
    
    def replace_placeholder_tokens(
        self,
        adaptor_dict: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        placeholder_values: Optional[List[dict]] = None,
        wp_encoder: Optional[nn.Module] = None,
    ):
        """
        函数功能:
            替换占位token
        它处理的占位主要有两种:
            第一种:文本里的特殊placeholder token
                这些token不是普通自然语言词,而是某些"结构化信息占位符",后面会用wp_encoder编码成embedding去替换
            第二种:<IMG_CONTEXT> token
                这是图像上下文占位token,后面会被视觉编码器输出的image feature替换
        """
        
        
        
        # 一、获取tokenizer
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        
        
        
        # 二、获取<IMG_CONTEXT> token的id
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        
        
        
        # 三、处理输入参数
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        
        
        
        # 四、输入embeding是空
        if inputs_embeds is None:




            # inputs_embeds = adaptor_dict['language_inputs']  # ❤️ 初始文本 embedding, 形状为[B,L,D] 其中B是batch size, L是文本长度, D是embedding维度
            
            # clone后得到非叶子张量，允许后续结构化placeholder替换参与正常梯度传播
            inputs_embeds = adaptor_dict['language_inputs'].clone()  # ❤️ 初始文本 embedding, 形状为[B,L,D] 其中B是batch size, L是文本长度, D是embedding维度
            
            
            
            
            input_ids = adaptor_dict['language__ids']        # ❤️ 对应的   token id,  形状为[B,L] 其中B是batch size, L是文本长度
            
            
            
            
            # 2a replace placeholder
            # 这一步的目的就是对input_ids进行处理,找到出现的所有的特殊id(为什么称之为特殊id,是因为它们是special token的映射),然后去掉重复的,最终赋值给special_ids
            smallest_added_id = self.tokenizer.additional_special_tokens_ids[0]  # 表示tokenizer中额外添加的特殊token的id列表的第一个id
            special_ids = torch.tensor(list(set(input_ids[(input_ids >= smallest_added_id)].tolist())), device=input_ids.device)
            special_ids = special_ids.view(-1, 1, 1)  # ❤️ 形状为[K,1,1] 其中K是这个batch里出现过的特殊token的种类数
            # print(f"Batch has special token ids: {special_ids.squeeze().tolist()}")  # 打印这个batch里出现过的特殊token的id
            batch_size, seq_len = input_ids.shape

            if special_ids.size(0) > 0 and len(placeholder_values) > 0:
                # special_ids.size(0) > 0,也就是这个 batch 至少出现过一种特殊 token.
                # len(placeholder_values) > 0,说明外部真的传进来了 placeholder 的内容,而不是一个空列表. 只有这两者都满足,我们才有必要进行下面的替换操作.
                
                """
                对于每个样本里出现的某个 special token,找到它第一次出现的位置;
                然后从 placeholder_values 中取出这个 token 对应的坐标序列，送入 wp_encoder 得到向量序列，再把这段向量序列写回 inputs_embeds 的对应位置。
                """
                
                # 后面创建坐标张量时，要让它的数据类型和 wp_encoder 的参数类型一致
                wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype

                # Create a mask where the special_ids are located
                mask = input_ids == special_ids  # ❤️ mask.shape = [K, B, L]  mask 是一个中间产物

                # Convert the mask to float and use torch.cumsum to get cumulative sum along the sequence length dimension
                cumsum_mask = torch.cumsum(mask.float(), dim=2)  # ❤️

                # Create a mask to get the first occurrence by checking where cumsum is 1
                first_occurrence_mask = (cumsum_mask == 1) & mask  # ❤️ 构造“只保留第一次出现位置”的 mask  形状为 [K, B, L] 其中 K 是特殊 token 的种类数, B 是 batch size, L 是文本长度

                # Use torch.argmax to get the indices of the first occurrence
                first_occurrences = torch.argmax(first_occurrence_mask.float(), dim=2)   # ❤️ 返回1所在的位置索引, 形状为 [K, B] 其中 K 是特殊 token 的种类数, B 是 batch size.
                # swap the dimensions to get the batch and sequence length
                first_occurrences = first_occurrences.transpose(0, 1)    # ❤️ 形状变为 [B, K] 其中 B 是 batch size, K 是特殊 token 的种类数.

                # get coords from label.placeholder_values with batch and special_id as key
                special_token_pos = first_occurrences.nonzero()    # ❤️ 形状为 [N, 2] 其中 N 是这个 batch 里所有特殊 token 的总出现次数, 每行是一个坐标 (b_id, key_id) 分别表示这个特殊 token 在第 b_id 个样本里, 它是这个样本里第 key_id 种特殊 token.

                # key_id是特殊id的索引, b_di是本batch内样本的索引
                coords = [torch.tensor(placeholder_values[b_id][special_ids[key_id].item()], device=input_ids.device, dtype=wp_encoder_dtype) for key_id, b_id in zip(special_token_pos[:, 1], special_token_pos[:, 0])]  # 形状为 [N, coord_len] 其中 N 是这个 batch 里所有特殊 token 的总出现次数, coord_len 是这个特殊 token 对应的坐标序列的长度. 注意这里我们把坐标序列都放在一个列表里了, 因为不同的特殊 token 可能对应不同长度的坐标序列.
                # print("哈哈哈哈哈哈哈哈", coords)  # 是一个点[x,y]
                coords_length_org = [len(coord) for coord in coords]      # coords_length_org=[2], 形状为 [N] 其中 N 是这个 batch 里所有特殊 token 的总出现次数, 每个元素是对应的坐标序列的长度. 这个列表后面会用来把 wp_encoder 的输出拆开.
                # print("哈哈哈哈", coords_length_org)  # 2
                coords = torch.cat(coords)                                # 先拼接起来
                wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)    # 放入 wp_encoder 得到向量表示
                # print("呵呵呵",wp_embeds.shape)   # shape [2,289]
                wp_embeds = torch.split(wp_embeds, coords_length_org)     # 再拆开
                # print("wp_embeds shape after split:", [embed.shape for embed in wp_embeds])  # 每个元素的形状都是 [coord_len, token_embedding_dim]
                # wp_embeds shape after split: [torch.Size([2, 896])]

                first_occurrences_filtered = [first_occurrences[i] for i in special_token_pos[:, 0]]

                for i, (pos, first_occurrence) in enumerate(zip(special_token_pos, first_occurrences_filtered)):
                    start = first_occurrence[pos[1]]
                    end = start + coords_length_org[i]
                    inputs_embeds[pos[0], start:end] = wp_embeds[i]  # 使用wp_embeds替换掉inputs_embeds中对应位置的原始文本embedding

            # 2b. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                #pixel_values is not None,说明外部真的传进来了图像数据
                # 

                all_pixel_values = [pixel_values]  # 预留接口,后续可以支持多种类型的图像输入,比如不同摄像头的图像,或者同一摄像头在不同时间点的图像等. 目前先假设只有一种图像输入,所以直接把它放在一个列表里.
                    
                all_image_features = []  # 用于存每一路图像提取出来的特征
                all_feature_lens = []    # 冗余变量
                _, N_embed, C_embed = inputs_embeds.shape  # _是batch size, N_embed是token数, C_embed是embedding维度
                
                # 进入图像处理循环
                for pixel_values_tmp in all_pixel_values:
                    BS, T, NP, C, H, W = pixel_values_tmp.shape  # BS是batch size, T是时间维度(比如视频的帧数), NP是每个样本里图像的数量(比如不同摄像头的图像), C是图像的通道数, H和W是图像的高和宽.
                    assert T == 1, "Only one frame is supported for now"  # 重点:说明目前这段代码只支持单帧图像输入,如果 T > 1，直接报错
                    # for multi-frame support, we need to change the code here
                    
                    # 去掉时间维度,因为已经知道T==1了,所以时间维度没必要保留了
                    pixel_values_tmp = pixel_values_tmp.view(BS, NP, C, H, W)

                    if pixel_values_tmp.dim() == 5:
                        pixel_values_tmp = pixel_values_tmp.reshape(BS*NP, C, H, W) # 这里reshape了,是为了喂给视觉编码器extract_feature,因为视觉 backbone 一般都希望输入是标准图像 batch 形式[batch, channel, height, width]
                    elif pixel_values_tmp.dim() != 4:
                        # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                        raise ValueError(f"pixel_values of shape {pixel_values_tmp.shape}, expect to be of 4 or 5 dimensions")
                    


                    # image_features = self.model.extract_feature(pixel_values_tmp)  # 视觉网络
                    # image_features = image_features.reshape(-1, C_embed)           # 行数：总共多少个视觉 token 列数：每个 token 的特征维度                 
                    # all_image_features.append(image_features)  # 因为可能未来有多路图像输入，所以先存列表，最后再拼接起来









                    image_features = self.model.extract_feature(pixel_values_tmp)  # 视觉网络
                    # image_features:
                    # [BS * NP, 256, C_embed]
                    # 其中每个原始patch对应256个视觉token，也就是16×16的空间网格。

                    expected_num_patches = (
                        self.num_cameras
                        * self.num_patches_per_camera
                    )
                    assert NP == expected_num_patches, (
                        f"Expected {expected_num_patches} image patches "
                        f"from {self.num_cameras} cameras, but received {NP}"
                    )

                    num_image_patches, num_tokens_per_patch, feature_dim = (
                        image_features.shape
                    )

                    feature_grid_size = int(
                        num_tokens_per_patch ** 0.5
                    )
                    assert (
                        feature_grid_size * feature_grid_size
                        == num_tokens_per_patch
                    ), (
                        "The number of visual tokens in each patch must form "
                        f"a square grid, but received {num_tokens_per_patch}"
                    )

                    assert feature_dim == C_embed, (
                        f"Visual feature dimension {feature_dim} does not "
                        f"match language embedding dimension {C_embed}"
                    )

                    # [BS*NP, 256, C_embed]
                    # → [BS*NP, C_embed, 16, 16]
                    image_features = image_features.transpose(1, 2)
                    image_features = image_features.reshape(
                        num_image_patches,
                        feature_dim,
                        feature_grid_size,
                        feature_grid_size,
                    )

                    # 每个patch从16×16=256个token池化为4×8=32个token。
                    # 每个相机有2个patch，因此每个相机最终保留64个token。
                    image_features = F.adaptive_avg_pool2d(
                        image_features,
                        output_size=(
                            self.visual_pool_height,
                            self.visual_pool_width,
                        ),
                    )

                    # [BS*NP, C_embed, 4, 8]
                    # → [BS*NP, 32, C_embed]
                    image_features = image_features.flatten(2)
                    image_features = image_features.transpose(1, 2).contiguous()

                    # 按固定顺序恢复每个样本的视觉token：
                    # front、front_left、front_right、rear、rear_left、rear_right。
                    # 每个样本最终得到12×32=384个视觉token。
                    image_features = image_features.reshape(
                        BS,
                        NP * self.num_visual_tokens_per_patch,
                        C_embed,
                    )

                    all_image_features.append(image_features)  # 因为可能未来有多路图像输入，所以先存列表，最后再拼接起来














                vit_embeds = torch.cat(all_image_features, dim=0)  # 拼接所有视觉特征 最终得到vit_embeds.shape = [N_visual_total, C_embed]其中 N_visual_total 是所有图像 token 总数
                inputs_embeds = inputs_embeds.reshape(BS * N_embed, C_embed)
                input_ids = input_ids.reshape(BS * N_embed)
                selected = (input_ids == self.img_context_token_id)  # 实际上这也是一个mask



                num_selected_image_tokens = int(
                    selected.sum().item()
                )
                num_visual_features = vit_embeds.reshape(
                    -1,
                    C_embed,
                ).shape[0]

                assert (
                    num_selected_image_tokens
                    == num_visual_features
                ), (
                    "The number of <IMG_CONTEXT> tokens does not match "
                    "the number of pooled visual features: "
                    f"{num_selected_image_tokens} vs {num_visual_features}"
                )



                # try:
                #     inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C_embed)
                # except Exception as e:
                #     vit_embeds = vit_embeds.reshape(-1, C)
                #     print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, '
                #         f'vit_embeds.shape={vit_embeds.shape}')
                #     n_token = selected.sum()
                #     inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]




                # 将池化后的视觉特征整理为[视觉token数量, embedding维度]
                vit_embeds_flat = vit_embeds.reshape(
                    -1,
                    C_embed,
                ).to(
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )

                # 找到所有<IMG_CONTEXT>位置
                selected_indices = selected.nonzero(
                    as_tuple=False
                ).squeeze(1)

                assert selected_indices.numel() == vit_embeds_flat.size(0), (
                    "The number of <IMG_CONTEXT> positions does not match "
                    "the number of pooled visual features: "
                    f"{selected_indices.numel()} vs {vit_embeds_flat.size(0)}"
                )

                # 使用非原地index_copy替换图像token，
                # 避免对requires_grad的叶子张量或其view执行原地操作
                inputs_embeds = inputs_embeds.index_copy(
                    0,
                    selected_indices,
                    vit_embeds_flat,
                )




                inputs_embeds = inputs_embeds.reshape(BS, N_embed, C_embed)
                input_ids = input_ids.reshape(BS, N_embed)
            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass
            
            adaptor_dict['language_inputs'] = inputs_embeds  # 替换后的文本 embedding, 形状为[B,L,D] 其中B是batch size, L是文本长度, D是embedding维度
            start_id = adaptor_dict['perm'][:,0]
            
            for b, i in enumerate(start_id):
                adaptor_dict['inputs'][b][:len(adaptor_dict['language_inputs'][b])-i] = inputs_embeds[b][i:]
            
        return adaptor_dict
            