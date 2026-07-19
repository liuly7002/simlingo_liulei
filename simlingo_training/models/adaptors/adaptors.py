from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from simlingo_training.utils.custom_types import DrivingExample


def cross_track_error(points: Tensor, path: Tensor):
    """
    Computes the cross track error between a set of points and a path.

    Args:
        points: The set of points to compute the cross track error for with shape [b, n, 2].
        path: The path to compute the cross track error with with shape [b, m, 2]. The path
            can contain nan values which indicates that the path is not available for that position.

    Returns:
        The cross track error for each point in the set of points with shape [b, n].
    """

    points, path = points.float(), path.float()

    ind = torch.arange(path.size(0), device=path.device)[:, None]
    closest = torch.cdist(points, path).nan_to_num_(torch.inf).argmin(-1)
    pt0 = path[ind, (closest - 1).clamp_min(0)]
    pt1 = path[ind, closest]
    pt2 = path[ind, (closest + 1).clamp_max(path.size(1) - 1)]

    tangent = (pt2 - pt1).nan_to_num_(0.0) + (pt1 - pt0).nan_to_num_(0.0)
    normal = torch.stack((tangent[..., 1], -tangent[..., 0]), dim=-1)
    normal = normal / normal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-2)

    return (points - pt1).mul(normal).sum(-1).abs()


class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
    

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """

    # B: batch size
    # N: waypoint 数量
    # 2: 每个 waypoint 的坐标维度, (x, y)
    # token_size: 输出的 token 的特征维度, 也就是最终要替换到文本里去的 embedding 的维度. 这个值必须要和语言模型的 embedding 维度一致.
    
    def __init__(
        self, token_size: int = 258, hidden_size: int = 64, hidden_size2: int = 128, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer     # 可选归一化层, 如果提供了就用它来归一化输入, 没有就不归一化直接送入MLP

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size), 
            nn.ReLU(True), 
            nn.Linear(hidden_size, hidden_size2), 
            nn.ReLU(True), 
            nn.Linear(hidden_size2, token_size)
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, N, 2]

        Returns:
            Output with dims [B, N, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)   # 对每个waypoint进行独立编码, 这个模块不是在建模轨迹序列关系,而是在做点级别embedding
        return x


class DrivingAdaptor(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim=256, predict_route_as_wps=False, speed_wps_mode=False,):
        
        super().__init__()
        
        self.heads = {}  # 用于保存不同任务对应的预测头
        self.order = []  # 用于记录任务的顺序

        ################## 0. 预测waypoints的两种方式 ##################
        
        # 方式1:使用自车真正的waypoints(执行,2d)
        self.speed_wps_mode = speed_wps_mode
        # 方式2:使用route作为waypoints,请注意这里的route是离散的参考路径(执行)
        self.predict_route_as_wps = predict_route_as_wps



        ################## 1. route 预测分支 ##################
        
        if predict_route_as_wps:  # 执行
            self.future_waypoints = 20  # 表示未来 route 一共预测 20 个 waypoint

            # 🚨 准备20个"位置槽位",每个槽位未来负责预测一个waypoint 形状为[1,20,hidden_size]
            self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, self.future_waypoints, hidden_size)))
            
            # route 预测头  输入是每个 query 对应的 feature，输出是一个二维增量  形状[B,20,2]
            self.route_head = nn.Sequential(nn.Linear(hidden_size, mlp_dim*2), nn.SiLU(True),nn.Linear(mlp_dim*2, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, 2, bias=False))
            
            self.queries = {'route': self.query_embeds_wps}
            self.sizes = {'route': self.future_waypoints}  # 记录每个任务对应有多少个 query
            self.heads["route"] = self.route_head
            self.order.append('route')


        
        ################## 2. speed wps 预测分支 ##################
        if speed_wps_mode == '2d':  # 执行这个
            dim = 2
        elif speed_wps_mode == '1d':
            dim = 1
        else:
            raise ValueError(f"speed_wps_mode must be '1d' or '2d', not {speed_wps_mode}")
        self.future_speed_waypoints = 10 #TODO: read from config    # 表示 speed_wps 分支预测 10 个未来点
        
        # 🚨 准备10个"位置槽位",每个槽位未来负责预测一个waypoint 形状为[1,10,hidden_size]
        self.query_embeds_speed = nn.Parameter(0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size)))
        # speed wps 预测头
        self.speed_wps_head = nn.Sequential(nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, dim, bias=False))

        self.heads["speed_wps"] = self.speed_wps_head
        self.queries['speed_wps'] = self.query_embeds_speed
        self.sizes['speed_wps'] = self.future_speed_waypoints
        self.order.append('speed_wps')


    def forward(self, driving_example: DrivingExample,**kwargs) -> Dict[str, Tensor]:

        """
        根据 batch size 构造该 adaptor 的输入 query 序列和 mask
        """

        try:
            driving_input = driving_example.driving_input
        except AttributeError:
            driving_input = driving_example
        
        b = driving_input.camera_images.shape[0]
        inputs = None

        # 扩展并拼接"位置槽位",形成inputs,形状[B,30,hidden_size]
        for input_type in self.order:
            query_embed = self.queries[input_type]
            if inputs is None:
                inputs = query_embed.expand(b, -1, -1)  # 如果当前还是第一次拼接，就先把 query 从 (1, N, D) 扩展成 (B, N, D)
            else:
                inputs = torch.cat((inputs, query_embed.expand(b, -1, -1)), dim=1)

        # 构造mask,形状为[B,30],这是全为true的布尔mask,表示所有的query token都是有效token,没有padding
        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {"inputs": inputs, "inputs_mask": inputs_mask}  # "inputs"：拼接好的 query embedding   "inputs_mask"：对应的有效位置 mask

    def get_predictions(self, features: Tensor,logits: Optional[Tensor] = None) -> Dict:

        """
        把主网络输出的 adaptor feature 切分开，再送入各自的 head, 得到最终预测结果
        """

        current_index = 0  # 记录当前切片起始位置,因为多个任务的 query 是拼接在一起的,所以 feature 也拼接在一起
        predictions = {}   # 字典,保存最终预测结果
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]

            feature = features[:, current_index: current_index + size]
            prediction = self.heads[input_type](feature).cumsum(1)  # .cumsum(1)表示沿着时间/序列维度做类加和,这说明网络head实际预测的是相邻点之间的增量,而最终结果要通过累加变成轨迹点序列

            predictions[input_type] = prediction
            current_index += size   # 切片起点向后移动，为下一个任务做准备
        
        return predictions


    def compute_loss(self, adaptor_features: Tensor, adaptor_logits: Tensor, _inputs: Dict[str, Tensor], example: DrivingExample) -> Dict[str, Tuple[Tensor, Tensor]]:
        
        """
        根据 adaptor_features 解码出预测结果，再和 label 对比，计算损失
        adaptor_features 是主网络的输出
        """

        label = example.driving_label
        assert label is not None
        
        if self.predict_route_as_wps:  # 执行
            label_route = label.path   #
        else:
            label_route = None

        # if self.speed_wps_mode == '2d':  # 执行
        #     label_speed_wps = label.waypoints[:, : self.future_waypoints + 1]  # 取前21个点(0-20) 可能是标签中包含当前点+未来20个点
        # elif self.speed_wps_mode == '1d':
        #     label_speed_wps = label.waypoints_1d
        # else:
        #     label_speed_wps = None
        if self.speed_wps_mode == '2d':
            label_speed_wps = label.waypoints[:, : self.future_speed_waypoints]
        elif self.speed_wps_mode == '1d':
            label_speed_wps = label.waypoints_1d
        else:
            label_speed_wps = None

        current_index = 0
        loss_dict = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]
            features_tmp = adaptor_features[:,current_index:current_index + size]  # 切出当前任务对应的特征,形状(B, size, hidden_size)
            label = locals()[f'label_{input_type}']

            prediction = self.heads[input_type](features_tmp).cumsum(1)
            
            loss = F.smooth_l1_loss(prediction, label, reduction="none").sum(-1)  # 计算Smooth L1 损失,这是一种介于L1和L2之间的损失,在轨迹预测中很常见
            # sum(-1): 对最后一个维度求和,所以二维点(x,y)的损失会加起来,一维点就相当于保留原值
            
            # if input_type == 'waypoints' and self.predict_route_as_wps:
            #     # compute cross track error
            #     cte = cross_track_error(prediction, label_waypoints)
            #     loss_dict[f"{input_type}_cte_loss"] = (cte, torch.ones_like(cte, dtype=torch.long))

            loss_dict[f"{input_type}_loss"] = (loss, torch.ones_like(loss, dtype=torch.long))
            loss_dict[f"{input_type}_prediction"] = prediction
            loss_dict[f"{input_type}_label"] = label
            current_index += size

        return loss_dict


class LanguageAdaptor(nn.Module):

    def __init__(self, language_model):  # 参数 "language_model" 是外部传进来的大模型, 这里是 InternVL2-1B
        
        super().__init__()
        
        self.embed_tokens = language_model.model.embed_tokens   # 取出 LLM 的 embedding lookup 表, 作用: token id → embedding 向量
        
        ############### 模型的输出头 ###############
        # 情况A(最常见):GPT/LLaMA结构,输出层叫lm_head
        if hasattr(language_model.model, "lm_head"):
            self.lm_head = language_model.model.lm_head
        # 情况B(有些模型):输出层叫embed_out
        elif hasattr(language_model.model, "embed_out"):
            self.lm_head = language_model.model.embed_out
        # 情况C(更底层结构)
        elif hasattr(language_model.model.base_model.model, 'output'):
            self.lm_head = language_model.model.base_model.model.output
        # 👉 如果模型没有输出头 → 报错
        else:  
            raise ValueError("Language model must have `lm_head` or `embed_out` attribute.")


    def forward(self, example: DrivingExample, inference=False, **kwargs) -> Dict[str, Tensor]:
        
        # 获取数据
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
            
        # 获取batch size
        b = driving_input.camera_images.size(0) # camera_images 形状为[BS, T, 2, 3, 448, 448]  这里是为了获取 batch size
        
        # 加载数据中的prompt
        if inference:  # 推理的时候加载,因为prompt_inference只包含问题不包含答案(推理执行)
            label = driving_input.prompt_inference
        else:          # 训练的时候加载,prompt既包括问题也包括答案(训练执行)
            label = driving_input.prompt
            
        # 取出label中的数据
        if label is not None:
            ids = label.phrase_ids.long()   # token id  形状[B,L]
            ids_valid = label.phrase_valid  # true => is fed into model   👉 输入 mask, 标记哪些 token 不是 padding (不是的位置为true), 形状为[B,L]
            ids_mask = label.loss_masking   # true => takes part in loss  👉 loss mask, 哪些 token 参与 loss, 直接把有效token的位置作为mask 形状为[B,L]

        # 核心,利用 token id 在 LLM 的 embedding lookup 表中查找对应的 embedding 向量
        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))   # [B,L,D]  实现 ids->embedding
        return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}

    def compute_loss(self, adaptor_features: Tensor, adaptor_logits: Tensor, inputs: Dict[str, Tensor], example: DrivingExample) -> Dict[str, Tuple[Tensor, Tensor]]:
        # adaptor_features：中间特征
        # adaptor_logits  ：模型输出
        # inputs          ：forward 的输出

        # del example  # 节省显存

        # # 如果没给 logits 用lm_head生成
        # if adaptor_logits is None:
        #     adaptor_logits = self.lm_head(outputs[:, :-1])  # 👉 outputs[:, :-1] 去掉最后一个 token outputs = LLM forward 的 hidden states（最后一层输出）
        # else:
        #     adaptor_logits = adaptor_logits[:, :-1]         # 👉 adaptor_logits[:, :-1] 如果已有 logits，也裁掉最后一位
        
        # # 在标准 LLM 中，流程是：

        # # input_ids / inputs_embeds
        # #         ↓
        # # Transformer（多层 attention）
        # #         ↓
        # # outputs（hidden states）   ← 就是这里的 outputs
        # #         ↓
        # # lm_head（线性层）
        # #         ↓
        # # logits（预测每个词的概率）
        
        
        # # 找出来用于计算loss的labels
        # labels = torch.where(inputs["_ids_mask"], inputs["_ids"], -1)  # mask=True → 用真实 token  mask=False → 设为 -1（ignore）
        
        # # Shift by 1 for next token prediction
        # # 👉 标准语言模型训练：
        # # input:  x1 x2 x3
        # # target:    x2 x3 x4
        # # 即：预测“下一个 token”
        # labels = labels[:, 1:]  # 丢掉index=0
        
        # # 计算交叉熵
        # language_loss = F.cross_entropy(adaptor_logits.flatten(0, -2), labels.flatten(), ignore_index=-1, reduction="none").view_as(labels)

        # return {"language_loss": (language_loss, labels.ne(-1))}  # language_loss: 每个 token loss    labels.ne(-1): 有效 mask

        del example  # 节省显存

        # ids_mask=True的位置才参与语言损失。
        labels = torch.where(
            inputs["_ids_mask"],
            inputs["_ids"],
            -1,
        )

        # 标准next-token prediction：
        # 第i个hidden feature预测第i+1个token。
        labels = labels[:, 1:]
        features_for_prediction = adaptor_features[:, :-1]

        valid_mask = labels.ne(-1)

        if valid_mask.any():
            # 只选取真正参与语言监督的位置。
            # 图像token、问题token、padding和Driving query均不会生成词表logits。
            selected_features = features_for_prediction[
                valid_mask
            ]
            selected_labels = labels[valid_mask]

            if adaptor_logits is None:
                selected_logits = self.lm_head(
                    selected_features
                )
            else:
                selected_logits = adaptor_logits[
                    :, :-1
                ][valid_mask]

            selected_loss = F.cross_entropy(
                selected_logits,
                selected_labels,
                reduction="none",
            )

            # 恢复为与labels相同的形状，
            # 以保持后续loss汇总逻辑不变。
            valid_indices = (
                valid_mask.reshape(-1)
                .nonzero(as_tuple=False)
                .squeeze(1)
            )

            language_loss_flat = torch.zeros(
                labels.numel(),
                device=selected_loss.device,
                dtype=selected_loss.dtype,
            )

            language_loss_flat = language_loss_flat.scatter(
                0,
                valid_indices,
                selected_loss,
            )

            language_loss = language_loss_flat.view_as(
                labels
            )
        else:
            # 极端情况下该batch没有语言监督位置，
            # 保留一条与模型特征相连的零梯度计算图。
            language_loss = (
                features_for_prediction.sum(dim=-1)
                * 0.0
            )

        return {
            "language_loss": (
                language_loss,
                valid_mask,
            )
        }




class AdaptorList(nn.Module):
    """
    Each adaptor is responsible for converting a driving example
    to a sequence of tokens and computing the loss on the token outputs.
    Adaptors are only used during training.
    每个适配器负责将一个driving example转换为一系列tokens,并计算基于tokens输出的损失。适配器仅在训练期间使用。
    """

    def __init__(self,driving: Optional[DrivingAdaptor] = None,language: Optional[LanguageAdaptor] = None,):
        super().__init__()
        self.driving = driving
        self.language = language

    @property
    def adaptors(self):
        """
        输出(👉 顺序是 language → driving):
        dct = 
        {
            "language": LanguageAdaptor,
            "driving": DrivingAdaptor
        }
        """
        dct: Dict[str, Adaptor] = {}
        if self.language is not None:
            dct["language"] = self.language
        if self.driving is not None:
            dct["driving"] = self.driving
        return dct

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]: 
        """
        Construct input embeddings for the given driving example.
        为给定的 driving example 构造输入 embedding
        """

        input_dict: Dict[str, Tensor] = {}
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = adaptor.forward(example, **kwargs)
            inputs_list.append(adaptor_input_dict["inputs"])            # language: [B, L, D]  driving: [B, 30, D]
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])  # language: [B, L]     driving: [B, 30]
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()})

        inputs = torch.cat(inputs_list, dim=1)           # [B, L+30, D]
        inputs_mask = torch.cat(inputs_mask_list, dim=1) # [B, L+30]
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list])  # [L,30]
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]  # 形状 [B, 1]
        # arange =
        # [
        #   [0],
        #   [1],
        #   ...
        #   [B-1]
        # ]

        # Apply random permutation of modalities during training
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)  # [B, L+30]
        # rand_perm =
        # [
        #   [0,1,2,3,...,L+30-1],
        #   [0,1,2,3,...,L+30-1],
        #   ...
        #   [0,1,2,3,...,L+30-1]
        # ]
        # Apply permutation to move invalid tokens to end of sequence
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)  # .byte()把布尔值转成 0/1：True -> 1 False -> 0
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]             # 👉 重排 token embedding[B, L+30, D]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]   # 👉 mask 同步重排 [B, L+30]
        input_dict["perm"] = perm                               # 👉 保存 permutation（用于恢复顺序）
        input_dict["split_sizes"] = split_sizes                 # 👉 保存切分信息
        return input_dict
        """
        input_dict = 
        {
            "inputs": [B, L+N, D],
            "inputs_mask": [B, L+N],
            "perm": 用于恢复顺序,
            "split_sizes": 用于拆分
        }
        """

    def compute_loss(
        self, features: Tensor, logits: Tensor, input_dict: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Distributes the output embeddings from the transformer to
        the correct loss function and returns a dictionary of losses.
        
        把 Transformer 的输出 "features(compute_loss函数的输入)" 分发给正确的 adaptor,让它们各自算各自的损失
        """

        # # 按 adaptor 拆分输出特征
        # features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features)
        # logits_by_adaptor = self.split_outputs_by_adaptor(input_dict, logits)
        # """
        # 把总的 features: [B, L+N, D] 拆成:
        # {
        #     "language": [B, L, D],
        #     "driving": [B, N, D]
        # }
        # """



        features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features,)
        # Driving分支不需要logits；
        # 语言分支将在有效答案位置根据hidden features局部计算logits。
        if logits is None:
            logits_by_adaptor = {
                key: None
                for key in self.adaptors.keys()
            }
        else:
            logits_by_adaptor = self.split_outputs_by_adaptor(
                input_dict,
                logits,
            )



        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}

        # Compute loss in each adaptor
        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}
        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = _gather_from_dict(input_dict, key + "_")
            adaptor_features = features_by_adaptor[key]# feature的值
            adaptor_logits = logits_by_adaptor[key]    # logits的值
            losses = adaptor.compute_loss(adaptor_features, adaptor_logits, adaptor_input_dict, example)
            loss_dict.update(losses)

        return loss_dict
        """
        loss_dict = 
        {
            "language_loss": ...,
            "route_loss": ...,
            "speed_wps_loss": ...
        }
        """

    def split_outputs_by_adaptor(self, input_dict: Dict[str, Tensor], outputs: Tensor) -> Dict[str, Tensor]:
        """
        Splits the output tensor into the correct output for each adaptor, according to the
        split_sizes in the input_dict.

        按照 split_sizes,把总输出切成各 adaptor 对应的输出段
        """
        # First reverse permutation
        inv_perm = input_dict["perm"].argsort(-1)  # 求逆排列,返回的是索引值,可以通过该索引值将input_dict["perm"]
        arange = torch.arange(inv_perm.size(0), device=inv_perm.device)[:, None]  # [B, 1]
        outputs = outputs[arange, inv_perm]  # 这一步之后，outputs 的 token 顺序就从“重排后的顺序”恢复成了“原始拼接顺序”

        # Now split output for each adaptor
        split_sizes = [int(x) for x in input_dict["split_sizes"]]  # [L,30]
        outputs_list = list(outputs.split(split_sizes, dim=1))     # 按长度切分
        return {key: outputs_list[i] for i, key in enumerate(self.adaptors.keys())}
        """
        {
            "language": outputs_list[0],
            "driving": outputs_list[1]
        }
        """


def _gather_from_dict(d: Dict[str, Tensor], prefix: str):
    out: Dict[str, Tensor] = {}  # dict comprehensions with if not supported
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out