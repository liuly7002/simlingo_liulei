import torch
from torch import nn
from typing import List, Optional
from transformers import AutoModel

class LingoInternVLModel(nn.Module):
    def __init__(self, variant, *args, **kwargs):
        super().__init__()
        # 加载模型
        self.model = AutoModel.from_pretrained(variant, trust_remote_code=True)  # trust_remote_code=True 表示允许 HuggingFace 执行模型仓库里自定义的 Python 代码
        # 语言模型原始词表一共有多少 token embedding
        try:
            self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
        except:
            self.num_embeddings = self.model.language_model.vocab_size
        self.use_global_img = None  # 预留接口
        self.processor = None       # 预留接口
        
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
        
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            # for_inputs_embeds_ids = input_ids.clone()
            # for_inputs_embeds_ids[(input_ids >= self.num_embeddings)] = 0
            # inputs_embeds = language_model.model.get_input_embeddings()(for_inputs_embeds_ids)
            inputs_embeds = adaptor_dict['language_inputs']  # 初始文本 embedding
            input_ids = adaptor_dict['language__ids']        # 对应的 token id  形状为[B,L] 其中L是文本长度，B是batch size
            
            # 2a replace placeholder
            smallest_added_id = self.tokenizer.additional_special_tokens_ids[0]  # 表示tokenizer中额外添加的特殊token的id列表的第一个id
            # special_ids 表示本 batch 出现过三种特殊 token, 例如 special_ids=tensor([32000, 32001, 32002])
            special_ids = torch.tensor(list(set(input_ids[(input_ids >= smallest_added_id)].tolist())), device=input_ids.device)
            # special_ids = torch.tensor(list(set(ids[(ids > 50294)].tolist())), device=ids.device)
            special_ids = special_ids.view(-1, 1, 1)   #   第 0 维：第几种 special token, 第 1 维：batch 中第几个样本, 第 2 维：序列中的第几个位置
            batch_size, seq_len = input_ids.shape

            if special_ids.size(0) > 0 and len(placeholder_values) > 0:
                # special_ids.size(0) > 0,也就是这个 batch 至少出现过一种特殊 token.
                # len(placeholder_values) > 0,说明外部真的传进来了 placeholder 的内容,这里我认为实际上placeholder_values就是传进来的waypoint
                
                """
                对于每个样本里出现的某个 special token,找到它第一次出现的位置;
                然后从 placeholder_values 中取出这个 token 对应的坐标序列，送入 wp_encoder 得到向量序列，再把这段向量序列写回 inputs_embeds 的对应位置。
                """
                
                # 后面创建坐标张量时，要让它的数据类型和 wp_encoder 的参数类型一致
                wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype

                # Create a mask where the special_ids are located
                # mask表示对于每一种 special token，在 batch 中每个样本的每个位置，它是否出现
                mask = input_ids == special_ids  # mask.shape = [K, B, L]

                # Convert the mask to float and use torch.cumsum to get cumulative sum along the sequence length dimension
                cumsum_mask = torch.cumsum(mask.float(), dim=2)

                # Create a mask to get the first occurrence by checking where cumsum is 1
                first_occurrence_mask = (cumsum_mask == 1) & mask  # 构造“只保留第一次出现位置”的 mask

                # Use torch.argmax to get the indices of the first occurrence
                first_occurrences = torch.argmax(first_occurrence_mask.float(), dim=2)
                # swap the dimensions to get the batch and sequence length
                first_occurrences = first_occurrences.transpose(0, 1)

                # get coords from label.placeholder_values with batch and special_id as key
                special_token_pos = first_occurrences.nonzero()

                coords = [torch.tensor(placeholder_values[b_id][special_ids[key_id].item()], device=input_ids.device, dtype=wp_encoder_dtype) for key_id, b_id in zip(special_token_pos[:, 1], special_token_pos[:, 0])]
                coords_length_org = [len(coord) for coord in coords]
                coords = torch.cat(coords)
                wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)    # 放入 wp_encoder 得到向量表示
                wp_embeds = torch.split(wp_embeds, coords_length_org)

                first_occurrences_filtered = [first_occurrences[i] for i in special_token_pos[:, 0]]

                for i, (pos, first_occurrence) in enumerate(zip(special_token_pos, first_occurrences_filtered)):
                    start = first_occurrence[pos[1]]
                    end = start + coords_length_org[i]
                    inputs_embeds[pos[0], start:end] = wp_embeds[i]

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                all_pixel_values = [pixel_values]
                    
                all_image_features = []
                all_feature_lens = []
                _, N_embed, C_embed = inputs_embeds.shape
                
                for pixel_values_tmp in all_pixel_values:
                    BS, T, NP, C, H, W = pixel_values_tmp.shape
                    assert T == 1, "Only one frame is supported for now"
                    # for multi-frame support, we need to change the code here
                    
                    pixel_values_tmp = pixel_values_tmp.view(BS, NP, C, H, W)

                    if pixel_values_tmp.dim() == 5:
                        pixel_values_tmp = pixel_values_tmp.reshape(BS*NP, C, H, W)
                    elif pixel_values_tmp.dim() != 4:
                        # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                        raise ValueError(f"pixel_values of shape {pixel_values_tmp.shape}, expect to be of 4 or 5 dimensions")
                    
                    image_features = self.model.extract_feature(pixel_values_tmp)
                    image_features = image_features.reshape(-1, C_embed)
                                        
                    all_image_features.append(image_features)

                vit_embeds = torch.cat(all_image_features, dim=0)
                inputs_embeds = inputs_embeds.reshape(BS * N_embed, C_embed)
                input_ids = input_ids.reshape(BS * N_embed)
                selected = (input_ids == self.img_context_token_id)
                try:
                    inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C_embed)
                except Exception as e:
                    vit_embeds = vit_embeds.reshape(-1, C)
                    print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, '
                        f'vit_embeds.shape={vit_embeds.shape}')
                    n_token = selected.sum()
                    inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]
                inputs_embeds = inputs_embeds.reshape(BS, N_embed, C_embed)
                input_ids = input_ids.reshape(BS, N_embed)
            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass
            
            adaptor_dict['language_inputs'] = inputs_embeds
            start_id = adaptor_dict['perm'][:,0]
            
            for b, i in enumerate(start_id):
                adaptor_dict['inputs'][b][:len(adaptor_dict['language_inputs'][b])-i] = inputs_embeds[b][i:]
            
        return adaptor_dict
            