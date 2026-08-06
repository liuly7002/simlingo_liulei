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

    # B: batch size
    # N: target points 数量,一般是2个target points
    # 2: 每个导航点的坐标维度, (x, y)
    # token_size: 输出的 token 的特征维度, 也就是最终要替换到文本里去的 embedding 的维度. 这个值必须要和语言模型的 embedding 维度一致.
    
    def __init__(self, token_size: int = 258, hidden_size: int = 64, hidden_size2: int = 128, norm_layer: Optional[nn.Module] = None):
        
        # 调用父类__init__()函数
        super().__init__()

        # 隐状态尺寸
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer     # 不进行归一化, 可选归一化层, 如果提供了就用它来归一化输入, 没有就不归一化直接送入MLP

        self.mlp = nn.Sequential(
            
            # 第一层线性层 2 -> 256
            nn.Linear(2, hidden_size), 
            nn.ReLU(True),

            # 第二层线性层  256 -> 512
            nn.Linear(hidden_size, hidden_size2), 
            nn.ReLU(True),

            # 第三层线性层 512 -> 语言模型的embedding的维度D
            nn.Linear(hidden_size2, token_size)
            )

    def forward(self, x: Tensor) -> Tensor:

        # 可选项非空的话,先对输入的点进行归一化
        if self.norm_layer is not None:
            x = self.norm_layer(x)

        x = self.mlp(x)   # 对每个 Target point 进行独立编码, 这个模块不是在建模轨迹序列关系,而是在做点级别embedding
        
        return x


class DrivingAdaptor(nn.Module):
    """
    为 route 和自车未来 waypoints 准备一组可学习的 query token;
    这些 query 经过联合 Transformer 后,再由各自预测头解码成二维轨迹,并与真实标签计算损失.

    在训练流程中承担三个阶段的工作：

        阶段1: forward()
        生成30个可学习Driving query
        [B,30,D]

        阶段2: 联合Transformer
        让30个query读取视觉、语言、导航和交互信息
        [B,30,D] → [B,30,D]

        阶段3: get_predictions() / compute_loss()
        前20个query → route
        后10个query → speed_wps
    """

    def __init__(self, hidden_size: int, mlp_dim=256, predict_route_as_wps=False, speed_wps_mode=False,):
        
        # hidden_size 表示每个 Driving query 的特征维度, 也必须与语言模型的隐藏维度一致

        super().__init__()
        
        self.heads = {}  # 用于保存不同任务对应的预测头
        self.order = []  # 用于记录任务的顺序

        ################## 0. 预测两类waypoints ##################
        
        # 1. 预测自车未来waypoints是预测的二维的点信息,也就是xy
        self.speed_wps_mode = speed_wps_mode

        # 2. 是否预测参考路径 True
        self.predict_route_as_wps = predict_route_as_wps



        ####################################### 1. route 预测分支 #######################################
        
        if predict_route_as_wps:  # 执行
            
            self.future_waypoints = 20  # 表示一共预测 20 个 route 点

            # 🚨 创建 route query (很小的随机数), 形状[1,20,D]
            self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, self.future_waypoints, hidden_size)))
            
            # route 预测头 每一个隐状态都需要接这个预测头才能预测该隐状态对应的输出
            # 输入是每个 query 对应的 feature，形状为[D], 输出是一个二维增量,表示相邻route点之间的而为增量Δx、Δy  形状[2] 形状变化 D->512->256->2
            self.route_head = nn.Sequential(nn.Linear(hidden_size, mlp_dim*2), nn.SiLU(True),nn.Linear(mlp_dim*2, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, 2, bias=False))
            
            self.queries = {'route': self.query_embeds_wps}# query [1,20,D]
            self.sizes = {'route': self.future_waypoints}  # 记录每个任务对应有多少个 query
            self.heads["route"] = self.route_head          # 预测头
            self.order.append('route')                     # 顺序


        
        ####################################### 2. speed wps 预测分支 #######################################
        if speed_wps_mode == '2d':  # 执行这个
            dim = 2
        elif speed_wps_mode == '1d':
            dim = 1
        else:
            raise ValueError(f"speed_wps_mode must be '1d' or '2d', not {speed_wps_mode}")
        
        self.future_speed_waypoints = 10 #TODO: read from config    # 表示 speed_wps 分支预测 10 个未来点
        
        # 🚨 创建 speed query (很小的随机数), 形状[1,10,D]
        self.query_embeds_speed = nn.Parameter(0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size)))
        
        # speed wps 预测头
        # 输入是每个 query 对应的 feature，形状为[hidden_size], 输出是一个二维增量,表示相邻speed_waypoint点之间的而为增量Δx、Δy  形状[2]形状变化 hidden_size -> 256 -> 2
        self.speed_wps_head = nn.Sequential(nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, dim, bias=False))

        self.heads["speed_wps"] = self.speed_wps_head        # 预测头 
        self.queries['speed_wps'] = self.query_embeds_speed  # query [1,10,hidden_size]
        self.sizes['speed_wps'] = self.future_speed_waypoints# 记录每个任务对应有多少个 query
        self.order.append('speed_wps')                       # 顺序


    # 获取 query 和对应的mask(这里全为1) 这些 query 将作为语言 Transformer 的输入
    def forward(self, driving_example: DrivingExample,**kwargs) -> Dict[str, Tensor]:

        """
        根据 batch size,把可学习 query 扩展并拼接,形成送入 Transformer 的 Driving query 序列
        """

        try:
            # 加载batch中用于输入的数据 这里只是为了通过图像获得batch size的大小
            driving_input = driving_example.driving_input
        except AttributeError:
            driving_input = driving_example
        
        # batch size 
        b = driving_input.camera_images.shape[0]
        inputs = None

        # 扩展并拼接"query",形成inputs,形状[B,30,D]
        # inputs[:,0:20]  → route query   inputs[:,20:30] → speed_wps query
        for input_type in self.order:
            
            query_embed = self.queries[input_type]
            
            if inputs is None:
                inputs = query_embed.expand(b, -1, -1)  # 将[1,20,D]扩展为[B,20,D] 将[1,10,D]扩展为[B,10,D]
            else:
                inputs = torch.cat((inputs, query_embed.expand(b, -1, -1)), dim=1)

        # 构造有效mask,形状为[B,30],inputs_mask全为true,表示所有的query token都是有效token,没有padding
        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {
                "inputs": inputs,          # [B,30,D]
                "inputs_mask": inputs_mask # [B,30]
                }

    def get_predictions(self, features: Tensor,logits: Optional[Tensor] = None) -> Dict:

        """
        把 Transformer 特征变成轨迹
        """

        current_index = 0  # 记录当前切片起始位置,因为 route 和 speed 的 query 是拼接在一起的,所以 feature 也拼接在一起
        predictions = {}   # 字典,保存最终预测结果

        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]

            # 切分特征 第一次循环feature = features[:, 0:19] 第二次循环feature = features[:, 20:30]
            feature = features[:, current_index: current_index + size]

            # 放入预测头 第一次循环[B,20,D]->[B,20,2] 第二次循环[B,10,D]->[B,10,2]
            prediction = self.heads[input_type](feature).cumsum(1)  # .cumsum(1)表示沿着时间/序列维度做类加和,这说明网络head实际预测的是相邻点之间的增量,而最终结果要通过累加变成轨迹点序列

            predictions[input_type] = prediction

            current_index += size   # 切片起点向后移动，为下一个任务做准备
        

        """
        {
            "route": prediction,       # [B,20,2] 表示预测的route 每个点表示在自车坐标系下的位置
            "speed_wps": prediction,   # [b,10,2] 表示预测的waypoits 每个点表示在自车坐标系下的位置
        }
        """
        return predictions

    def compute_loss(self, adaptor_features: Tensor, adaptor_logits: Tensor, _inputs: Dict[str, Tensor], example: DrivingExample) -> Dict[str, Tuple[Tensor, Tensor]]:
        
        """
        这个函数执行两件事
            1. 用预测头将Transformer特征解码成轨迹
            2. 将预测轨迹和真实轨迹计算Smooth L1损失
        """

        # 读取batch
        label = example.driving_label
        assert label is not None
        
        
        # route 标签
        if self.predict_route_as_wps:  # 执行
            label_route = label.path   # 自车坐标系下的原始参考路径点(40个,每个都是[x,y]两点之间相距1m)
        else:
            label_route = None


        # speed waypoints 标签
        if self.speed_wps_mode == '2d':
            label_speed_wps = label.waypoints[:, : self.future_speed_waypoints]  # 2D形式：自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y](无几何增强)
        elif self.speed_wps_mode == '1d':
            label_speed_wps = label.waypoints_1d
        else:
            label_speed_wps = None

        current_index = 0
        loss_dict = {}
        for i, input_type in enumerate(self.order):

            # 20 10
            size = self.sizes[input_type]

            # 分任务切出route和speed waypoints 的 Transformer 特征
            features_tmp = adaptor_features[:,current_index:current_index + size]  # 切出当前任务对应的特征,形状(B, size, hidden_size)
            
            # label_route label_speed_wps
            label = locals()[f'label_{input_type}']

            # 放入预测头 第一次循环[B,20,D]->[B,20,2] 第二次循环[B,10,D]->[B,10,2]
            prediction = self.heads[input_type](features_tmp).cumsum(1)
            
            loss = F.smooth_l1_loss(prediction, label, reduction="none").sum(-1)  # 计算Smooth L1 损失,这是一种介于L1和L2之间的损失,在轨迹预测中很常见
            # sum(-1): 对最后一个维度求和,所以二维点(x,y)的损失会加起来,一维点就相当于保留原值
        

            loss_dict[f"{input_type}_loss"] = (loss, torch.ones_like(loss, dtype=torch.long))
            loss_dict[f"{input_type}_prediction"] = prediction
            loss_dict[f"{input_type}_label"] = label
            current_index += size

        return loss_dict


class LanguageAdaptor(nn.Module):
    """
    LanguageAdaptor 的作用可以概括为两部分:
        进入 Transformer 前: 把文本 token ID 转换成语言 embedding
        经过 Transformer 后: 把语言 hidden feature 转换成词表 logits,并计算下一个 token 的语言生成损失

    它相当于语言任务与整个多模态网络之间的接口:
        文本token ID
            ↓ LanguageAdaptor.forward()
        文本embedding
            ↓ 图像和TARGET_POINT特征替换
            ↓ 联合Transformer
        语言hidden feature
            ↓ LanguageAdaptor.compute_loss()
        语言生成loss
    """

    
    # 取得语言模型的两个核心模块 -> embedding层和词表输出头
    def __init__(self, language_model):  # 参数 "language_model" 是外部传进来的大模型, 这里是 InternVL2-1B
        
        super().__init__()
        
        # 1. 取出语言模型的 embedding 词表, 作用: token id → embedding 向量
        self.embed_tokens = language_model.model.embed_tokens  
        
        # 2. 取出语言模型的输出头 作用: D维隐状态 -> 整个词表的logits

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


    # 将 token id 转换为 embedding, 这些 embedding 将作为语言 Transformer 的输入
    def forward(self, example: DrivingExample, inference=False, **kwargs) -> Dict[str, Tensor]:
        
        # 获取batch输入网络的数据
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
            
        # 获取batch size
        b = driving_input.camera_images.size(0) # camera_images 形状为[BS, T, 12, 3, 448, 448]  这里是为了获取 batch size
        
        # 加载数据中的prompt
        if inference:  # 推理的时候加载,因为prompt_inference只包含问题不包含答案(推理执行)
            label = driving_input.prompt_inference
        else:          # 训练的时候加载,prompt既包括问题也包括答案(训练执行)  这里主要是token id了
            label = driving_input.prompt
            
        # 取出label中的数据
        if label is not None:
            ids = label.phrase_ids.long()   # token id  形状[B,L]
            ids_valid = label.phrase_valid  # true => is fed into model   👉 输入 mask, 标记哪些 token 不是 padding (不是的位置为true), 形状为[B,L]
            ids_mask = label.loss_masking   # true => takes part in loss  👉 loss mask, 哪些 token 参与 loss, 直接把有效token的位置作为mask 形状为[B,L]

        # 核心,利用 token id 在语言模型的 embedding 词表中查找对应的 embedding 向量 [B,L,D]
        # 此时 <IMG_CONTEXT> 对应位置也只是先从 embedding 表取得一个普通 embedding
        # 同理，<TARGET_POINT> 位置也会被实际目标点坐标 embedding 替换
        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))   # [B,L,D]  实现 ids->embedding
        
        return {
                "inputs": inputs,         # [B,L,D]  文本token的embedding
                "inputs_mask": ids_valid, # [B,L]    哪些token不是padding、可以进入Transformer
                "_ids": ids,              # [B,L]    原始token ID，作为语言真实标签
                "_ids_mask": ids_mask     # [B,L]    哪些token需要计算语言loss
                }


    # 根据Transformer输出计算语言loss(显存优化版)
    def compute_loss(self, adaptor_features: Tensor, adaptor_logits: Tensor, inputs: Dict[str, Tensor], example: DrivingExample) -> Dict[str, Tuple[Tensor, Tensor]]:
        # adaptor_features：中间特征
        # adaptor_logits  ：模型输出
        # inputs          ：forward 的输出

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
            selected_features = features_for_prediction[valid_mask]
            selected_labels = labels[valid_mask]

            if adaptor_logits is None:
                selected_logits = self.lm_head(selected_features)
            else:
                selected_logits = adaptor_logits[:, :-1][valid_mask]

            selected_loss = F.cross_entropy(selected_logits,selected_labels,reduction="none",)

            # 恢复为与labels相同的形状，
            # 以保持后续loss汇总逻辑不变。
            valid_indices = (valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1))

            language_loss_flat = torch.zeros(labels.numel(),device=selected_loss.device,dtype=selected_loss.dtype,)

            language_loss_flat = language_loss_flat.scatter(0,valid_indices,selected_loss,)

            language_loss = language_loss_flat.view_as(labels)

        else:
            # 极端情况下该batch没有语言监督位置，
            # 保留一条与模型特征相连的零梯度计算图。
            language_loss = (features_for_prediction.sum(dim=-1) * 0.0)

        return {
            "language_loss": (
                language_loss,
                valid_mask,
            )
        }




class AdaptorList(nn.Module):

    def __init__(self,driving: Optional[DrivingAdaptor] = None,language: Optional[LanguageAdaptor] = None,):

        # 调用父类__init__()函数
        super().__init__()
        
        # driving adaptor
        self.driving = driving
        
        # language adaptor
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

        # example: 一个batch的数据

        input_dict: Dict[str, Tensor] = {}

        # 列表, 用于存放 language 和 driving 各自的 embedding 及对应的 mask, 最终用于拼接
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        # 获得 language 和 driving 的 embedding
        for key, adaptor in self.adaptors.items():

            # 1. 先调用 forward 函数, 为的就是获得 language 和 driving 分别的 embedding
            adaptor_input_dict = adaptor.forward(example, **kwargs)
            
            # 2. 将 forward 函数输出的"inputs"拿出来 堆叠为列表  这是输入语言 Transformer 的 embedding
            inputs_list.append(adaptor_input_dict["inputs"])            # language: [B, L, D]  driving: [B, 30, D]
            
            # 3. 将 forward 函数输出的"inputs_mask"拿出来 堆叠为列表 这表示哪些位置输入语言Transformer
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])  # language: [B, L]     driving: [B, 30]
            
            
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()})

        # 拼接 language embedding 和 driving query 以形成网络整体的输入 embedding 顺序为 [L个语言embedding | 20个route query | 10个speed_wps query]
        inputs = torch.cat(inputs_list, dim=1)           # [B, L+30, D]
        
        # 拼接以形成网络整体输入对应的 mask 表示哪些位置需要输入 哪些位置不用输入
        inputs_mask = torch.cat(inputs_mask_list, dim=1) # [B, L+30]

        # 获取 language 的长度和 driving 数据的长度
        # 这个信息后面用于将语言Transformer的总输出重新拆解成 language输出[B,L,D] driving输出[B,30,D]
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list])  # split_sizes=[L,30]
        
        # 这些内容是对通过[B,L+30]的inputs_mask对[B, L+30, D]的inputs进行重新排序,使得原始顺序[padding | padding | 有效语言 | Driving query]变为[有效语言 | Driving query | padding | padding]
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]  # 形状 [B, 1]
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)  # [B, L+30]
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)  # .byte()把布尔值转成 0/1：True -> 1 False -> 0
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]             # 👉 重排后的 embedding [B, L+30, D]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]   # 👉 mask 同步重排 [B, L+30]
        input_dict["perm"] = perm                               # 👉 保存 permutation（用于恢复顺序）
        input_dict["split_sizes"] = split_sizes                 # 👉 保存切分信息[L,30]
        
        
        return input_dict


    def compute_loss(self, features: Tensor, logits: Tensor, input_dict: Dict[str, Tensor], example: DrivingExample) -> Dict[str, Tuple[Tensor, Tensor]]:


        features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features,)
        
        # Driving分支不需要logits；
        # 语言分支将在有效答案位置根据hidden features局部计算logits。
        if logits is None:
            logits_by_adaptor = {key: None for key in self.adaptors.keys()}
        else:
            logits_by_adaptor = self.split_outputs_by_adaptor(input_dict,logits,)



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