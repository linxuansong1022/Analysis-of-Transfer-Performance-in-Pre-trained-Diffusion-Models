import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .model_1d import UNet1D

# 定义一个继承自 nn.Module 的 Block 类，用于构建 Transformer 中的一个块
class Block(nn.Module):
    # 初始化方法，接收输入维度 dim、头的数量 num_heads、MLP 扩展比例 mlp_ratio、qkv 偏置标志 qkv_bias 和丢弃率 dropout 作为参数
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout=0.0):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 定义第一个 Layer Normalization 层，对输入的最后一个维度进行归一化，维度为 dim
        self.norm1 = nn.LayerNorm(dim)
        # 定义多头注意力层，输入维度为 dim，头的数量为 num_heads，丢弃率为 dropout，是否使用偏置由 qkv_bias 决定
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, bias=qkv_bias)
        # 定义第二个 Layer Normalization 层，同样对输入的最后一个维度进行归一化，维度为 dim
        self.norm2 = nn.LayerNorm(dim)
        # 计算 MLP 隐藏层的维度，为输入维度 dim 乘以扩展比例 mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 定义一个顺序模块，包含一个线性层、GELU 激活函数、丢弃层、另一个线性层和丢弃层
        self.mlp = nn.Sequential(
            # 第一个线性层，输入维度为 dim，输出维度为 mlp_hidden_dim
            nn.Linear(dim, mlp_hidden_dim),
            # GELU 激活函数
            nn.GELU(),
            # 丢弃层，丢弃率为 dropout
            nn.Dropout(dropout),
            # 第二个线性层，输入维度为 mlp_hidden_dim，输出维度为 dim
            nn.Linear(mlp_hidden_dim, dim),
            # 丢弃层，丢弃率为 dropout
            nn.Dropout(dropout)
        )

    # 前向传播方法，接收输入张量 x
    def forward(self, x):
        # 注释说明输入 x 的形状为 (批次大小 B, 序列长度 L, 特征维度 D)
        # x shape: (B, L, D)
        # 对输入 x 进行归一化，然后通过注意力块，最后与原始输入 x 相加，实现残差连接
        x = x + self._attention_block(self.norm1(x))
        # 对 x 进行第二次归一化，然后通过 MLP 层，最后与 x 相加，实现残差连接
        x = x + self.mlp(self.norm2(x))
        # 返回处理后的张量 x
        return x

    # 定义一个私有方法，用于处理注意力块的计算
    def _attention_block(self, x):
        # 注释说明输入 x 的形状从 (B, L, D) 转换为 (L, B, D)，因为 nn.MultiheadAttention 的输入要求序列长度维度在前
        # x shape: (B, L, D) -> (L, B, D)
        # 交换 x 的第一个和第二个维度
        x = x.transpose(0, 1)
        # 通过多头注意力层进行计算，忽略注意力权重
        x, _ = self.attn(x, x, x)
        # 注释说明 x 的形状从 (L, B, D) 转换回 (B, L, D)
        # x shape: (L, B, D) -> (B, L, D)
        # 再次交换 x 的第一个和第二个维度
        return x.transpose(0, 1)

# 定义一个继承自 nn.Module 的 AttentionHead 类，用于创建神经网络的注意力头
class AttentionHead(nn.Module):
    """
    Class to create an attention head for a neural network.
    The layer string is a comma separated string of layer definitions.
    The layer definitions are of the form:

    Use_CLS_Token:bool:dim
    Pool:k:s
    Reshape
    Insert_CLS_Token
    Block:dim:num_heads:mlp_ratio:num_blocks
    Extract_CLS_Token
    Dense:in_dim:out_dim:act

    """
    # 初始化方法，接收一个层定义字符串 layer_string 作为参数
    def __init__(self, layer_string):
        # 调用父类 nn.Module 的初始化方法
        super(AttentionHead, self).__init__()
        # 保存层定义字符串
        self.layer_string = layer_string
        # 调用 build_model 方法构建模型层
        self.layers = self.build_model()

        # 根据层定义字符串解析是否使用 CLS 标记和维度
        self.use_cls_token, dim = self.parse_use_cls_token(layer_string.split(',')[0].strip())
        # 如果使用 CLS 标记
        if self.use_cls_token:
            # 定义一个可训练的 CLS 标记参数，初始值为全零，形状为 (1, 1, dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

    # 前向传播方法，接收输入张量 x
    def forward(self, x):
        # 遍历模型的每一层
        for layer in self.layers:
            # 将输入 x 通过当前层进行处理
            x = layer(x)
        # 返回处理后的张量 x
        return x

    # 定义一个方法，用于根据层定义字符串构建模型层
    def build_model(self):
        # 初始化一个空列表，用于存储模型层
        layers = []
        # 将层定义字符串按逗号分割成多个层定义
        layer_definitions = self.layer_string.split(',')

        # 遍历每个层定义
        for layer_definition in layer_definitions:
            # 去除层定义字符串两端的空格
            layer_definition = layer_definition.strip()
            # 如果层定义以 'Attention' 开头
            if layer_definition.startswith('Attention'):
                # 解析层定义，获取维度 dim、头的数量 num_heads、MLP 扩展比例 mlp_ratio 和块的数量 num_blocks
                dim, num_heads, mlp_ratio, num_blocks = self.parse_block_layer(layer_definition)
                # 循环创建指定数量的 Block 块
                for i in range(num_blocks):
                    # 创建一个 Block 块并添加到 layers 列表中
                    layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))
            # 如果层定义以 'Flatten' 开头
            elif layer_definition.startswith('Flatten'):
                # 添加一个 Flatten 层到 layers 列表中
                layers.append(nn.Flatten())
            # 如果层定义以 'Dense' 开头
            elif layer_definition.startswith('Dense'):
                # 解析层定义，获取输入维度 in_dim、输出维度 out_dim 和激活函数类型 activation
                in_dim, out_dim, activation = self.parse_dense_layer(layer_definition)
                # 添加一个线性层到 layers 列表中
                layers.append(nn.Linear(in_dim, out_dim))
                # 如果激活函数类型为 'relu'
                if activation == 'relu':
                    # 添加一个 ReLU 激活函数层到 layers 列表中
                    layers.append(nn.ReLU())
                # 如果激活函数类型为 'sigmoid'
                elif activation == 'sigmoid':
                    # 添加一个 Sigmoid 激活函数层到 layers 列表中
                    layers.append(nn.Sigmoid())
                # 如果激活函数类型为 'tanh'
                elif activation == 'tanh':
                    # 添加一个 Tanh 激活函数层到 layers 列表中
                    layers.append(nn.Tanh())
                # 如果激活函数类型为 'identity'，不做任何操作
                elif activation == 'identity':
                    pass
            # 如果层定义以 'Pool' 开头
            elif layer_definition.startswith('Pool'):
                # 解析层定义，获取池化核大小 pool_size 和步长 stride
                pool_size, stride = self.parse_pool_layer(layer_definition)
                # 添加一个二维平均池化层到 layers 列表中
                layers.append(nn.AvgPool2d(pool_size, stride=stride))
            # 如果层定义以 'FullyPool' 开头
            elif layer_definition.startswith('FullyPool'):
                # 添加一个自定义的 Lambda 层，对输入在第一个维度上求均值
                layers.append(LambdaLayer(lambda x: x.mean(dim=1)))
            # 如果层定义以 'Use_CLS_Token' 开头，不做任何操作
            elif layer_definition.startswith('Use_CLS_Token'):
                pass
            # 如果层定义以 'Reshape' 开头
            elif layer_definition.startswith('Reshape'):
                # 添加一个自定义的 Lambda 层，对输入进行形状重塑
                layers.append(LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')))
            # 如果层定义以 'Insert_CLS_Token' 开头
            elif layer_definition.startswith('Insert_CLS_Token'):
                # 添加一个自定义的 Lambda 层，将 CLS 标记插入到输入的开头
                layers.append(LambdaLayer(lambda x: torch.cat((self.cls_token.to(x.device).expand(x.shape[0], -1, -1),
                                                               x), dim=1)))
            # 如果层定义以 'Extract_CLS_Token' 开头
            elif layer_definition.startswith('Extract_CLS_Token'):
                # 添加一个自定义的 Lambda 层，提取输入的第一个元素
                layers.append(LambdaLayer(lambda x: x[:, 0]))

        # 将 layers 列表中的层组合成一个顺序模块并返回
        return nn.Sequential(*layers)

    # 定义一个方法，用于解析 Use_CLS_Token 层定义
    def parse_use_cls_token(self, layer_definition):
        # 将层定义字符串按冒号分割成多个部分
        layer_definition = layer_definition.split(':')
        # 将第二个部分转换为布尔值，表示是否使用 CLS 标记
        use_cls_token = bool(layer_definition[1])
        # 将第三个部分转换为整数，表示维度
        dim = int(layer_definition[2])
        # 返回是否使用 CLS 标记和维度
        return use_cls_token, dim

    # 定义一个方法，用于解析 Pool 层定义
    def parse_pool_layer(self, layer_definition):
        # 将层定义字符串按冒号分割成多个部分
        layer_definition = layer_definition.split(':')
        # 将第二个部分转换为整数，表示池化核大小
        pool_size = int(layer_definition[1])
        # 将第三个部分转换为整数，表示步长
        stride = int(layer_definition[2])
        # 返回池化核大小和步长
        return pool_size, stride

    # 定义一个方法，用于解析 Block 层定义
    def parse_block_layer(self, layer_definition):
        # 将层定义字符串按冒号分割成多个部分
        layer_definition = layer_definition.split(':')
        # 将第二个部分转换为整数，表示维度
        dim = int(layer_definition[1])
        # 将第三个部分转换为整数，表示头的数量
        num_heads = int(layer_definition[2])
        # 将第四个部分转换为浮点数，表示 MLP 扩展比例
        mlp_ratio = float(layer_definition[3])
        # 将第五个部分转换为整数，表示块的数量
        num_blocks = int(layer_definition[4])
        # 返回维度、头的数量、MLP 扩展比例和块的数量
        return dim, num_heads, mlp_ratio, num_blocks

    # 定义一个方法，用于解析 Dense 层定义
    def parse_dense_layer(self, layer_definition):
        # 将层定义字符串按冒号分割成多个部分
        layer_definition = layer_definition.split(':')
        # 将第二个部分转换为整数，表示输入维度
        in_dim = int(layer_definition[1])
        # 将第三个部分转换为整数，表示输出维度
        out_dim = int(layer_definition[2])
        # 获取第四个部分，表示激活函数类型
        activation = layer_definition[3]
        # 返回输入维度、输出维度和激活函数类型
        return in_dim, out_dim, activation

# 定义一个继承自 nn.Module 的 LambdaLayer 类，用于创建自定义层
class LambdaLayer(nn.Module):
    # 初始化方法，接收一个 lambda 函数 lambd 作为参数
    def __init__(self, lambd):
        # 调用父类 nn.Module 的初始化方法
        super(LambdaLayer, self).__init__()
        # 保存 lambda 函数
        self.lambd = lambd

    # 前向传播方法，接收输入张量 x
    def forward(self, x):
        # 将输入 x 通过 lambda 函数进行处理并返回结果
        return self.lambd(x)

# 定义一个函数，用于从指定时间步提取系数，并将其重塑为 [batch_size, 1, 1, ...] 以便进行广播
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, ...] for broadcasting purposes.
    """
    # 获取时间步 t 所在的设备
    device = t.device
    # 根据时间步 t 从 v 中提取相应的系数，并将其转换为浮点数类型，然后移动到 t 所在的设备上
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # 将提取的系数重塑为 [batch_size, 1, 1, ...] 的形状
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

# 定义一个继承自 nn.Module 的 GaussianDiffusionTrainer1D 类，用于一维高斯扩散模型的训练
class GaussianDiffusionTrainer1D(nn.Module):
    # 初始化方法，接收模型 model、初始噪声系数 beta_1、最终噪声系数 beta_T 和总时间步数 T 作为参数
    def __init__(self, model, beta_1, beta_T, T):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 保存模型
        self.model = model
        # 保存总时间步数
        self.T = T

        # 注册一个缓冲区，用于存储从 beta_1 到 beta_T 均匀分布的噪声系数，数据类型为双精度
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # 计算 alpha 值，即 1 - beta
        alphas = 1. - self.betas
        # 计算累积的 alpha 值
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 注册缓冲区，用于存储累积 alpha 值的平方根
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        # 注册缓冲区，用于存储 1 - 累积 alpha 值的平方根
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    # 前向传播方法，接收输入张量 x_0
    def forward(self, x_0):
        # 随机生成时间步 t，范围从 0 到 T - 1，形状为 (x_0 的批次大小,)，设备与 x_0 相同
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # 生成与 x_0 形状相同的随机噪声
        noise = torch.randn_like(x_0)

        # 直接使用 3D 张量，无需维度调整
        # 根据公式计算 x_t，即当前时间步的输入
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        # 将 x_t 和时间步 t 输入到模型中，得到预测的噪声
        predicted_noise = self.model(x_t, t)
        # 计算预测噪声和真实噪声之间的均方误差损失，不进行降维
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        # 返回损失
        return loss

# 定义一个继承自 nn.Module 的 GaussianDiffusionTrainer1DReg 类，用于一维高斯扩散模型的回归训练
class GaussianDiffusionTrainer1DReg(nn.Module):
    # 初始化方法，接收模型 model、初始噪声系数 beta_1、最终噪声系数 beta_T、总时间步数 T 和分类时间步 classify_T 作为参数
    def __init__(self, model, beta_1, beta_T, T, classify_T):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 保存模型
        self.model = model
        # 保存总时间步数
        self.T = T
        # 保存分类时间步
        self.classify_T = classify_T

        # 注册一个缓冲区，用于存储从 beta_1 到 beta_T 均匀分布的噪声系数，数据类型为双精度
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # 计算 alpha 值，即 1 - beta
        alphas = 1. - self.betas
        # 计算累积的 alpha 值
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 注册缓冲区，用于存储累积 alpha 值的平方根
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        # 注册缓冲区，用于存储 1 - 累积 alpha 值的平方根
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        # 定义一个线性层，输入维度为 128，输出维度为 1
        self.fc = nn.Linear(128, 1)

    # 前向传播方法，接收输入张量 x_0 和高度标签 heights
    def forward(self, x_0, heights):
        # 创建一个形状为 (x_0.shape[0],) 的张量 t，其所有元素的值都为 classify_T，数据类型为 torch.long，设备与 x_0 相同
        t = torch.full(
                size=(x_0.shape[0],),
                fill_value=self.classify_T,
                dtype=torch.long,
                device=x_0.device
            )

        # 生成一个与 x_0 形状相同的高斯噪声张量
        noise = torch.randn_like(x_0)

        # 根据公式计算 x_t，即当前时间步的输入
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        # 调用模型的 get_multiple_features 方法，获取 x_t 在时间步 t 的特征，这里对应不同块
        features = self.model.get_multiple_features(x_t, t)

        # 对第一个特征在第三个维度（序列长度）上求均值
        features_processed = features[0].mean(dim=2)

        # 如果处理后的特征维度与线性层的输入维度不匹配，打印警告信息
        if features_processed.shape[1] != self.fc.in_features:
            print(f"Warning: Feature dimension ({features_processed.shape[1]}) "
                  f"does not match fc layer input dimension ({self.fc.in_features}). "
                  "Regression results might be incorrect.")

        # 将处理后的特征输入到线性层中，得到预测值
        pre = self.fc(features_processed)

        # 如果高度标签是一维的，将其扩展为二维
        if heights.ndim == 1:
            heights = heights.unsqueeze(1)

        # 计算预测值和高度标签之间的均方误差损失，不进行降维
        loss = F.mse_loss(pre, heights, reduction='none')
        # 返回损失
        return loss

# 定义一个继承自 nn.Module 的 GaussianDiffusionSampler1D 类，用于一维高斯扩散模型的采样
class GaussianDiffusionSampler1D(nn.Module):
    # 初始化方法，接收模型 model、初始噪声系数 beta_1、最终噪声系数 beta_T 和总时间步数 T 作为参数
    def __init__(self, model, beta_1, beta_T, T):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 保存模型
        self.model = model
        # 保存总时间步数
        self.T = T

        # 注册一个缓冲区，用于存储从 beta_1 到 beta_T 均匀分布的噪声系数，数据类型为双精度
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # 计算 alpha 值，即 1 - beta
        alphas = 1. - self.betas
        # 计算累积的 alpha 值
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 对累积的 alpha 值进行填充，使其长度与总时间步数相同
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # 注册缓冲区，用于存储 1 / alpha 的平方根
        self.register_buffer('coef1', torch.sqrt(1. / alphas))
        # 注册缓冲区，用于存储 coef1 * (1 - alpha) / sqrt(1 - 累积 alpha 值)
        self.register_buffer('coef2', self.coef1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        # 注册缓冲区，用于存储后验方差
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # 定义一个方法，用于根据当前时间步的输入 x_t、时间步 t 和预测的噪声 eps 预测上一个时间步的均值
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # 断言 x_t 和 eps 的形状相同
        assert x_t.shape == eps.shape
        # 根据公式计算上一个时间步的均值
        return (
            extract(self.coef1, t, x_t.shape) * x_t -
            extract(self.coef2, t, x_t.shape) * eps
        )

    # 定义一个方法，用于计算当前时间步的均值和方差
    def p_mean_variance(self, x_t, t):
        # 下面：仅在 KL 计算中使用对数方差
        # 拼接后验方差和噪声系数
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        # 根据时间步 t 提取相应的方差
        var = extract(var, t, x_t.shape)

        # 将 x_t 和时间步 t 输入到模型中，得到预测的噪声
        eps = self.model(x_t, t)
        # 根据预测的噪声计算上一个时间步的均值
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        # 返回上一个时间步的均值和方差
        return xt_prev_mean, var

    # 前向传播方法，接收初始噪声 x_T
    def forward(self, x_T):
        """
        Algorithm 2.
        """
        # 将初始噪声赋值给 x_t
        x_t = x_T
        # 从最后一个时间步开始，反向遍历到第一个时间步
        for time_step in reversed(range(self.T)):
            # 创建一个全为当前时间步的张量 t，形状为 (x_T 的批次大小,)，数据类型为长整型
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 计算当前时间步的均值和方差
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # 当时间步大于 0 时，添加随机噪声
            if time_step > 0:
                noise = torch.randn_like(x_t)
            # 当时间步为 0 时，不添加噪声
            else:
                noise = 0
            # 根据均值和方差更新 x_t
            x_t = mean + torch.sqrt(var) * noise
            # 断言 x_t 中不包含 NaN 值
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        # 将最后一个时间步的 x_t 赋值给 x_0
        x_0 = x_t
        # 将 x_0 的值裁剪到 [-1, 1] 范围内
        return torch.clip(x_0, -1, 1)

