
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch
from torch import nn
from einops import rearrange 
import math
from torchinfo import summary 

class AttentionHead(nn.Module):
    """ 
    用于创建神经网络注意力头的类。
    该类接受一个逗号分隔的层定义字符串，支持的层定义格式如下：
    
    Use_CLS_Token:bool:dim        # 是否使用CLS Token及其维度
    Pool:k:s                      # 池化层，k为池化核大小，s为步长
    Reshape                       # 重塑层，用于改变张量形状
    Insert_CLS_Token              # 插入CLS Token的层
    Block:dim:num_heads:mlp_ratio:num_blocks  # Transformer块，包含维度、注意力头数、MLP比率和块数
    Extract_CLS_Token             # 提取CLS Token的层
    Dense:in_dim:out_dim:act      # 全连接层，包含输入维度、输出维度和激活函数

    """
    
    def __init__(self, layer_string): #layer_string是层定义字符串，例如layer_string = f"Attention:256:8:4:{num_blocks},"，用来配置网络结构
        super(AttentionHead, self).__init__()
        self.layer_string = layer_string #保存层定义字符串
        self.layers = self.build_model() #调用build_model方法构建网络

        # Build CLS token based on layer string 
        self.use_cls_token, dim = self.parse_use_cls_token(layer_string.split(',')[0].strip())#解析第一个层定义，获取是否使用CLS Token和CLS Token的维度
        if self.use_cls_token:#如果需要使用CLS Token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))#创建一个形状为(1, 1, dim)的零张量，第一个1表示batch size，第二个1表示序列长度，dim表示特征维度
        #CLS Token是一个可学习的参数，通过自注意力机制，可以关注到所有的特征块，能学习到整张图片的全局信息，
        #在分类任务中，我们需要一个代表整张图片的向量
        #CLS Token 的输出正好可以作为这个全局表示
        #这个表示可以直接用于分类

        #网络的前向传播 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) #将输入x依次通过所有层 
        return x
    
    #build_model方法用于根据层定义字符串构建网络
    def build_model(self):
        layers = []
        layer_definitions = self.layer_string.split(',') #将层定义字符串按逗号分割成一个列表    

        for layer_definition in layer_definitions: #遍历每个层定义
            layer_definition = layer_definition.strip()#去除字符串两端的空白字符
            if layer_definition.startswith('Attention'):#如果层定义以'Attention'开头
                dim, num_heads, mlp_ratio, num_blocks = self.parse_block_layer(layer_definition)#解析层定义，获取维度、注意力头数、MLP比率和块数
                for i in range(num_blocks):#遍历每个块
                    layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))#添加指定数量的Transformer块
            elif layer_definition.startswith('Flatten'):#如果层定义以'Flatten'开头
                layers.append(nn.Flatten())#添加一个展平层
            elif layer_definition.startswith('Dense'):#如果层定义以'Dense'开头
                in_dim, out_dim, activation = self.parse_dense_layer(layer_definition)#解析层定义，获取输入维度、输出维度和激活函数
                layers.append(nn.Linear(in_dim, out_dim))#添加一个全连接层
                if activation == 'relu':#如果激活函数为'rel u'
                    layers.append(nn.ReLU())#添加ReLU激活函数
                elif activation == 'sigmoid':#如果激活函数为'sigmoid'
                    layers.append(nn.Sigmoid())#添加Sigmoid激活函数
                elif activation == 'tanh':#如果激活函数为'tanh'
                    layers.append(nn.Tanh())#添加Tanh激活函数
                elif activation == 'identity':#如果激活函数为'identity'
                    pass
            elif layer_definition.startswith('Pool'):#如果层定义以'Pool'开头    
                pool_size, stride = self.parse_pool_layer(layer_definition)#解析层定义，获取池化核大小和步长
                layers.append(nn.AvgPool2d(pool_size, stride=stride))#添加一个平均池化层
            elif layer_definition.startswith('FullyPool'):#如果层定义以'FullyPool'开头
                layers.append(LambdaLayer(lambda x: x.mean(dim=1)))#添加一个Lambda层，用于计算每个样本的平均值
            elif layer_definition.startswith('Use_CLS_Token'): #如果层定义以'Use_CLS_Token'开头
                pass #什么也不做
            elif layer_definition.startswith('Reshape'):#如果层定义以'Reshape'开头
                layers.append(LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')))#添加一个Lambda层，用于重塑张量形状
            elif layer_definition.startswith('Insert_CLS_Token'):#如果层定义以'Insert_CLS_Token'开头
                layers.append(LambdaLayer(lambda x: torch.cat((self.cls_token.to(x.device).expand(x.shape[0], -1, -1),
                                                               x), dim=1)))#添加一个Lambda层，用于在输入张量中插入CLS Token
            elif layer_definition.startswith('Extract_CLS_Token'):#如果层定义以'Extract_CLS_Token'开头
                layers.append(LambdaLayer(lambda x: x[:, 0]))#添加一个Lambda层，用于提取CLS Token

        return nn.Sequential(*layers)#将所有层组合成一个序列化的神经网络模型
    
    #解析CLS Token的配置，输入："Use_CLS_Token:True:256"，输出：(True, 256)，表示使用CLS Token，维度为256
    def parse_use_cls_token(self, layer_definition):
        layer_definition = layer_definition.split(':')# 将字符串按冒号分割
        use_cls_token = bool(layer_definition[1])# 第二个参数转换为布尔值，表示是否使用CLS Token
        dim = int(layer_definition[2])#第三个参数转换为整数，表示CLS Token的维度
        return use_cls_token, dim #返回是否使用CLS Token和其维度
    #解析池化层的配置，输入："Pool:2:2"，输出：(2, 2)，表示池化核大小为2x2，步长为2
    def parse_pool_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')#将字符串按冒号分割
        pool_size = int(layer_definition[1])#第一个参数转换为整数，表示池化核大小
        stride = int(layer_definition[2])#第二个参数转换为整数，表示步长
        return pool_size, stride#返回池化核大小和步长
    #解析Transformer块的配置，输入："Attention:256:8:4:1"
    #输出：(256, 8, 4.0, 1)，表示：
    #特征维度为256
    #8个注意力头
    #MLP比率为4.0
    #1个Transformer块
    def parse_block_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')#将字符串按冒号分割
        dim = int(layer_definition[1])#第一个参数转换为整数，表示维度
        num_heads = int(layer_definition[2])#第二个参数转换为整数，表示注意力头数
        mlp_ratio = float(layer_definition[3])#第三个参数转换为浮点数，表示MLP比率
        num_blocks = int(layer_definition[4])#第四个参数转换为整数，表示块数
        return dim, num_heads, mlp_ratio, num_blocks#返回维度、注意力头数、MLP比率和块数
    #解析全连接层的配置，输入"Dense:256:128:relu"
    #输出：(256, 128, 'relu')，表示：
    #输入维度为256
    #输出维度为128
    #激活函数为relu
    def parse_dense_layer(self, layer_definition):#解析全连接层的配置
        layer_definition = layer_definition.split(':')#将字符串按冒号分割
        in_dim = int(layer_definition[1])#第一个参数转换为整数，表示输入维度
        out_dim = int(layer_definition[2])#第二个参数转换为整数，表示输出维度
        activation = layer_definition[3]#第三个参数表示激活函数
        return in_dim, out_dim, activation#返回输入维度、输出维度和激活函数

#Lambda层，用于自定义操作，将简单的Python操作集成到神经网络中
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
#标准的Transformer Encoder Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # 构造函数，接收多个参数来配置这个块：
        # - dim: 输入和输出的特征维度 (dimensionality)
        # - num_heads: 多头注意力机制中的头数
        # - mlp_ratio: MLP 中间隐藏层维度相对于输入维度的比例 (默认是4倍)
        # - qkv_bias: 是否在计算 Q, K, V 时添加偏置项
        # - drop: MLP 和 Attention 投影层之后的 Dropout 概率
        # - attn_drop: Attention 权重计算后的 Dropout 概率
        # - drop_path: Stochastic Depth (DropPath) 的概率 (这里默认为0, 后面会看到被设为Identity)
        # - act_layer: MLP 中使用的激活函数 (默认是 GELU)
        # - norm_layer: 使用的归一化层 (默认是 LayerNorm)
        super().__init__()
        #第一个子模块：归一化层1
        self.norm1 = norm_layer(dim)
        # 创建第一个归一化层实例，通常放在多头注意力之前 (Pre-Normalization)
        # 第二个子模块：多头自注意力 (Multi-Head Self-Attention, MHSA)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        #
        self.drop_path = nn.Identity() # I just default set this to identity function
        #第四个子模块：归一化层2
        self.norm2 = norm_layer(dim)
        # 创建第二个归一化层实例，通常放在 MLP 之前
        # 计算 MLP 中间隐藏层的维度
        mlp_hidden_dim = int(dim * mlp_ratio)
         # 五个子模块：MLP，作为前馈网络
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)#创建 Mlp 模块实例，传入输入维度、计算出的隐藏层维度、激活函数和 Dropout 概率

    def forward(self, x):
        # 定义数据如何流过这个块
        # --- 第一个子层：Multi-Head Self-Attention ---
        # 1. 输入 x 首先通过第一个归一化层 (self.norm1)
        # 2. 归一化后的结果输入到自注意力模块 (self.attn)
        # 3. 注意力模块的输出通过 DropPath 层 (self.drop_path, 在这里是 Identity, 不起作用)
        # 4. 将处理后的结果与原始输入 x 相加 (这是第一个残差连接 Residual Connection)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # --- 第二个子层：MLP / Feed-Forward Network ---
        # 1. 上一步的结果 x 通过第二个归一化层 (self.norm2)
        # 2. 归一化后的结果输入到 MLP 模块 (self.mlp)
        # 3. MLP 模块的输出通过 DropPath 层 (self.drop_path, 在这里是 Identity, 不起作用)
        # 4. 将处理后的结果与上一步的输出 x 相加 (这是第二个残差连接 Residual Connection)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x #返回这个 Transformer Block 的最终输出


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # 构造函数，接收配置参数：
        # - in_features: 输入特征的维度
        # - hidden_features: 隐藏层的特征维度 (可选，默认为 in_features)
        # - out_features: 输出特征的维度 (可选，默认为 in_features)
        # - act_layer: 使用的激活函数类 (默认是 nn.GELU)
        # - drop: Dropout 的概率 (默认为 0，即不使用 Dropout)
        super().__init__()
        out_features = out_features or in_features## 如果未指定输出维度，则默认为输入维度
        hidden_features = hidden_features or in_features## 如果未指定隐藏层维度，则默认为输入维度
        self.fc1 = nn.Linear(in_features, hidden_features)## 创建第一个全连接层，将输入特征维度映射到隐藏层维度
        self.act = act_layer()## 创建激活函数实例，默认是GELU
        self.fc2 = nn.Linear(hidden_features, out_features)## 创建第二个全连接层，将隐藏层特征维度映射到输出维度
        self.drop = nn.Dropout(drop)## 创建Dropout层，用于防止过拟合

    def forward(self, x):
        x = self.fc1(x)## 将输入特征 x 通过第一个全连接层
        x = self.act(x)## 将第一个全连接层的输出通过激活函数
        x = self.drop(x)## 将激活函数的输出通过Dropout层
        x = self.fc2(x)## 将Dropout层的输出通过第二个全连接层
        x = self.drop(x)## 将第二个全连接层的输出通过Dropout层
        return x## 返回最终的输出

#标准的Transformer Attention Block
class Attention(nn.Module):
    # 构造函数，接收配置参数：
    def __init__(
            self,
            dim,
            num_heads=8,#注意力头的数量
            qkv_bias=False,#是否使用偏置
            qk_norm=False,#是否使用归一化
            attn_drop=0.,#注意力dropout概率
            proj_drop=0.,#投影dropout概率
            norm_layer=nn.LayerNorm,#归一化层类型
    ):
        super().__init__()
        assert dim % num_heads == 0 #确保维度可以被注意力头数整除
        self.num_heads = num_heads# 将注意力头数赋值给实例变量
        self.head_dim = dim // num_heads# 计算每个注意力头的维度
        self.scale = self.head_dim ** -0.5# 计算缩放因子，用于防止点积结果过大

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)# 创建一个线性层，将输入维度映射到3倍的输入维度，输入维度是 dim (C)，输出维度是 dim * 3 (因为 Q, K, V 每个的维度都是 dim)。
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()#如果 qk_norm 为 True，则创建一个 LayerNorm 层用于对 Q 进行归一化。
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()#同上，为 K 创建可选的归一化层。
        self.attn_drop = nn.Dropout(attn_drop) #创建一个 Dropout 层，用于在计算完 softmax 之后的注意力权重上应用。
        self.proj = nn.Linear(dim, dim)#创建一个线性层，将输入维度映射到输出维度，输入维度是 dim (C)，输出维度也是 dim (C)。
        self.proj_drop = nn.Dropout(proj_drop)#创建一个 Dropout 层，用于在最终的输出上应用。

    def forward(self, x):
        # 定义前向传播逻辑，输入 x 的形状预期为 (B, N, C)
        # B: batch size (批次大小)
        # N: sequence length (序列长度，例如图像块的数量 + CLS Token)
        # C: embedding dimension (特征维度，等于 self.dim)
        B, N, C = x.shape # 获取输入张量的形状信息
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # 1. self.qkv(x): 将输入 x 通过 qkv 线性层，输出形状为 (B, N, C*3)。
        # 2. .reshape(B, N, 3, self.num_heads, self.head_dim):
        #    将形状变为 (B, N, 3, H, Dh)。3 代表 Q, K, V。
        # 3. .permute(2, 0, 3, 1, 4):
        #    重排维度，将形状变为 (3, B, H, N, Dh)。
        #    这样做是为了方便后续将 Q, K, V 分开，并让每个头独立计算注意力。
        #    (B:批次, H:头, N:序列长度, Dh:头维度)
        q, k, v = qkv.unbind(0)#将 qkv 张量沿着第一个维度（索引为 0）进行拆分，得到三个新的张量 q, k, v，形状为 (B, H, N, Dh)。
        q, k = self.q_norm(q), self.k_norm(k)#对 Q 和 K 进行归一化处理。

        q = q * self.scale#缩放因子
        attn = q @ k.transpose(-2, -1)#计算注意力权重
        attn = attn.softmax(dim=-1)#对注意力权重进行归一化处理
        attn = self.attn_drop(attn)#应用 Dropout 层 
        x = attn @ v#计算加权和 

        x = x.transpose(1, 2).reshape(B, N, C)#重塑张量形状，将形状变为 (B, N, C)。
        x = self.proj(x)#将重塑后的张量通过线性层进行投影。
        x = self.proj_drop(x)#应用 Dropout 层   
        return x

def extract(v, t, x_shape):
    """
    根据输入的时间步 t 从系数张量 v 中提取相应的数值，
    并将提取出的数值（形状通常为 [batch_size]）重新塑形为与目标张量 x_shape 兼容的形状（例如 [batch_size, 1, 1, 1]），以便后续进行广播操作。
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


#标准DDPM训练器的实现
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):#model是U-Net模型，beta_1和beta_T是扩散过程的噪声系数初始和最终值，T是时间步长
        super().__init__()

        self.model = model #U-Net模型
        self.T = T #时间步长

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
            #生成方差序列，包含T个元素的一维张量，并转换成双精度浮点数
            #同时注册一个缓冲区，作为模型状态的一部分，会自动保存到模型中，不会参与梯度更新，命名betas
        alphas = 1. - self.betas  #定义alpha，beatas是噪声
        alphas_bar = torch.cumprod(alphas, dim=0)#是 alphas 的累积乘积，表示从初始状态到时间步 t 的累积噪声系数。
        #计算累积方差，用于后续的扩散过程
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        #分别是 alphas_bar 和 1 - alphas_bar 的平方根，用于后续的扩散计算。
        
    def forward(self, x_0):#模型的前向传播过程，实现了 DDPM 的训练算法
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        #t 是一个长度为 x_0.shape[0] 的张量，包含从 0 到 T-1 的随机整数，表示每个样本的时间步。
        noise = torch.randn_like(x_0) #noise 是一个与 x_0 形状相同的张量，包含从标准正态分布中采样的噪声
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        #其中，alphat_t是 sqrt_alphas_bar，epsilon是 noise。
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')#计算预测的噪声和实际噪声之间的均方误差损失
        return loss


class GaussianDiffusionClassifyTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, head, classify_T):#model是U-Net模型，beta_1和beta_T是扩散过程的噪声系数初始和最终值，T是时间步长，head是分类头，classify_T是分类时间步长
        super().__init__()
        self.head = head  #分类头
        self.classify_T = classify_T #分类时间步长，用于分类任务，选择特定时间步的特征进行分类
        self.model = model #U-Net模型
        self.loss_fn = torch.nn.CrossEntropyLoss() #交叉熵损失函数，用于分类任务
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        #全连接层
        if self.head=='fc':
            self.pool = nn.AdaptiveAvgPool2d((4, 4))#自适应平均池化层，将输入特征图的空间维度（高和宽）调整为精确的4*4，通道数不变
            self.fc = nn.Linear(256*4*4, 10) #穿管一个全连接层，输入特征数(不同的block和池化层特征数不同)映射到输出特征数（10），代表10个类别的得分（CIFAR10数据集）
        #多层感知机
        elif self.head=='mlp':
            self.pool = nn.AdaptiveAvgPool2d((4, 4))#自适应平均池化层，将输入特征图的空间维度（高和宽）调整为精确的4*4，通道数不变
            self.fc = nn.Sequential(#定义了一个构成MLP的层序列
                        nn.Linear(256*4*4, 256),#第一个线性层，输入出256个特征
                        nn.BatchNorm1d(256), #对第一个线性层的输出应用 1D 批归一化,稳定和加速训练过程，全连接层的输出是[batch_size，num_featuers]的二维张量，在num_featuers这个维度上进操作，计算均值和方差并进行归一化
                        nn.ReLU(inplace=True),#激活函数引入非线性,直接在原始输入张量（Tensor）的内存上进行计算并覆盖结果，而不是创建一个新的张量来存储结果。
                        nn.Linear(256, 64),# 第二个线性层，从 256 维降到 64 维
                        nn.BatchNorm1d(64), # 1D 批归一化
                        nn.ReLU(inplace=True), # ReLU 激活函数
                        nn.Linear(64, 10) # 输出线性层，从 64 维映射到 10 个类别得分
                    )  #78.20

        elif self.head=='attn':
            ### attention
            self.rer = LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c'))
            num_blocks = 1
            layer_string = f"Attention:256:8:4:{num_blocks}," # dim, num_heads, mlp_ratio, num_blocks 
            # # build model 
            self.fc = AttentionHead(layer_string)# 81e85
        elif self.head=='cnn':
            self.fc = nn.Sequential(
                        nn.Conv2d(256, 64, 3, 2),# # 输入通道256, 输出通道64, 卷积核大小3x3, 步长2
                        nn.BatchNorm2d(64),#对64个通道进行批归一化
                        nn.ReLU(inplace=True), #使用ReLU激活函数，inplace=True节省内存
                        nn.Conv2d(64, 10, 3, 2),#输入通道64, 输出通道10 (对应10个类别), 卷积核3x3, 步长2
                    )  #80e75


    def forward(self, x_0, labels):
        #print(x_0.shape,labels.shape) torch.Size([80, 3, 32, 32]) torch.Size([80])
        t = torch.full(
                size=(x_0.shape[0],), 
                fill_value=self.classify_T, 
                dtype=torch.long,  # 时间步应为整数类型
                device=x_0.device  # 保持与原张量相同的设备（如 GPU）
            )
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        #这几行的目的就是更具原始图像x_0和目标时间步self.classify_T，计算出对应的带噪图像x_t
        #前向加噪步骤不是为了训练UNet去噪（UNet是预训练好的，通常被冻结），而是作为一个必要的预处理步骤，用来生成特定噪声水平的图像 x_t。
        #这个 x_t 是后续从UNet中提取用于分类的特征所必需的输入。分类器实际上是在学习如何根据DDPM在特定噪声阶段的内部特征表示来进行分类。
        features = self.model.get_multiple_features(x_t, t)
        #得到x_t之后，才能将其和时间步一起输入到UNet的特征提取方法中，这样提取出来的features才是在目标噪声水平self.classify_T下的特征表示
        #print(len(features),features[0].shape)# [80, 256, 16, 16]
        #最后，这些特征被送入分类头进行处理，并与labels计算损失，从而训练分类器
        if self.head=='fc':
            #对 UNet 提取的特征图 features[0] 应用池化
            feature = self.pool(features[0])#.squeeze(-1).squeeze(-1)
            # 将池化后的特征图展平成一维向量，从维度1（通道维）开始展平，得到[B,8192]
            feature = torch.flatten(feature, start_dim=1)
            # 将展平的特征向量输入线性层，得到预测得分
            pre = self.fc(feature)

        elif self.head=='mlp':
            #应用池化
            feature = self.pool(features[0])#.squeeze(-1).squeeze(-1)
            #展平特征
            feature = torch.flatten(feature, start_dim=1)
            #将展平的特征输入整个 MLP 序列
            pre = self.fc(feature)

        elif self.head=='attn':
            feature = self.rer(features[0])
            # `features[0]` 是从 U-Net 中间层提取的特征图，形状类似于 [batch_size, channels, height, width] (例如 [80, 256, H, W])。
            # `self.rer` 是一个 LambdaLayer，它使用 einops.rearrange 将特征图重塑为序列形式：
            # 输入形状: [B, C, H, W]
            # 输出形状 (`feature`): [B, N, C]，其中 N = H * W (序列长度)。
            # 这一步是为了让特征图能作为 Transformer 注意力头的输入。
            
            # print(feature.shape) [80, 256, 1, 1]
            # feature = torch.flatten(feature, start_dim=1)
            # print(feature.shape)# [32, 65536]
            pre = self.fc(feature)
            # `self.fc` 在 'attn' 模式下是一个 AttentionHead 实例。
            # 它接收重塑后的序列 `feature` ([B, N, C]，例如 [80, H*W, 256]) 作为输入。
            # AttentionHead 内部通常包含 Transformer Block(s)，可能还使用了 CLS Token。
            # 它会对输入的序列进行自注意力计算和特征变换。
            # `AttentionHead` 的最终输出 (`pre`) 理论上应该将特征映射到类别得分。
            # 假设 AttentionHead 的最后一层是一个线性层，将特征维度 C 映射到 10 个类别。
            # 输出 `pre` 的形状可能是 [B, N, 10] (如果对序列中每个位置都预测) 或 [B, 10] (如果使用了 CLS Token 并提取了其输出)。
            pre = pre.mean(dim=1).squeeze(1)
            # `pre.mean(dim=1)`: 对 `pre` 的序列维度 (维度 1，长度为 N=H*W) 进行平均。
            #   - 如果 `pre` 的形状是 [B, N, 10]，这一步会将其平均为 [B, 10]。它计算了序列中所有位置的平均类别得分。
            #   - 如果 `pre` 的形状已经是 [B, 10] (例如，AttentionHead 内部提取了 CLS Token)，那么 `.mean(dim=1)` 会引发错误或产生意外结果，因为维度 1 不存在或不是序列维度。但从代码看，它预期输入是 [B, N, C']。
            # `.squeeze(1)`: 移除大小为 1 的维度。如果 `.mean(dim=1)` 之后结果是 [B, 10]，这一步通常没有效果。如果结果是 [B, 1, 10]，它会变成 [B, 10]。鉴于 `.mean(dim=1)` 已经移除了维度1，这一步很可能是多余的或为了处理某些特殊情况。
            # 最终 `pre` 的形状预期是 [batch_size, 10]，代表了每个样本的 10 个类别预测得分。
    
        ### 4.CNN
        elif self.head=='cnn':
            feature = self.fc(features[0])#将从 U-Net 提取的特征输入到 CNN 头中
            pre = feature.mean(dim=3).mean(dim=2)#全局平均池化，对每个通道和整个空间维度H和W都进行了平均
        loss = self.loss_fn( pre, labels)

        return pre,loss

