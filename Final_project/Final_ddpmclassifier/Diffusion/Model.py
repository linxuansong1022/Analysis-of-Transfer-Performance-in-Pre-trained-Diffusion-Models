import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

#swish激活函数定义，Resnet中使用，和tail中使用
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#将时间步t转换为时间嵌入向量
class TimeEmbedding(nn.Module):
    """
    TimeEmbedding类的主要作用是将扩散模型中的时间步t转换为高维的时间嵌入向量。这在扩散模型中起着至关重要的作用:

    1. 时间信息的表示:
    - 扩散过程是一个逐步添加噪声的过程,每个时间步t对应不同的噪声水平
    - 通过将离散的时间步t转换为连续的高维向量,模型可以更好地理解和利用时间信息
    - 使用正弦和余弦函数进行编码,可以捕捉时间步之间的相对关系

    2. 条件控制:
    - 时间嵌入向量作为条件信息注入到U-Net的各个层中
    - 使模型能够根据不同的时间步t采用不同的去噪策略
    - 帮助模型理解在去噪过程中应该去除多少噪声

    3. 技术细节:
    - 使用位置编码的思想,将时间步编码为正弦和余弦信号
    - 通过多层感知机进行进一步转换,增强表达能力
    - 使用Swish激活函数增加非线性
    """
    def __init__(self, T, d_model, dim):#T是总的时间部署，d_model表示时间嵌入的维度，必须是偶数，因为后续会拆分成正弦和余弦，dim是最终输出的时间嵌入向量纬度
        assert d_model % 2 == 0 #检查是否是偶数
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)#freq_i= (2i /d_model)*ln(10000)
        #这里如果d_model=8，会生成一个张量 [0, 2, 4, 6]，之所以使用步长为 2，是因为后续会将生成的位置编码分为正弦和余弦两部分，这样可以确保每个位置都有对应的正弦和余弦分量。
        #将生成的张量中的每个元素除以 d_model，目的是对生成的位置索引进行归一化，将其缩放到 [0, 1) 的范围内，避免数值过大。
        #乘以 math.log(10000)，这一步是为了调整频率的尺度。在位置编码中，不同的频率可以表示不同的时间步信息，通过这种方式可以让模型学习到不同时间步之间的相对关系。
        emb = torch.exp(-emb)#w_i=exp(-freq_i)
        #对 emb 中的每个元素取负后再进行指数运算。这样做是为了将之前计算得到的频率值转换为合适的范围，使得后续的正弦和余弦函数能够在不同的频率上振荡，从而为不同的时间步提供不同的编码。
        pos = torch.arange(T).float()#0 到 T-1 的浮点数张量 pos，表示不同的时间步。
        emb = pos[:, None] * emb[None, :]# 将 pos 和 emb 进行矩阵乘法，得到一个形状为 (T, d_model) 的张量。
        #得到PE(t,i)=t*w_i
        assert list(emb.shape) == [T, d_model // 2] #检查emb的形状是否符合预期
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        #计算得到最终的位置编码PE(t,2i)=sin(t*1/10000^(2i/d_model)),PE(t,2i+1)=cos(t*1/10000^(2i/d_model))
        #将 emb 中的每个元素分别取正弦和余弦值，并将结果堆叠在一起，形成一个形状为 (T, d_model // 2, 2) 的张量。
        assert list(emb.shape) == [T, d_model // 2, 2]#检查emb的形状是否符合预期
        emb = emb.view(T, d_model)#将 emb 展平成一个形状为 (T, d_model) 的张量。
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),#将 emb 转换为 nn.Embedding 类型，并使用 from_pretrained 方法初始化,创建的这个 Embedding 层的查找表
            #接收一个已经准备好的张量 (Tensor) emb 作为参数，并将这个 emb 的内容 直接复制到 nn.Embedding 内部的查找表（权重矩阵）中。
            nn.Linear(d_model, dim),#将 emb 展平成一个形状为 (T, d_model) 的张量。
            Swish(),#使用 Swish 激活函数。
            nn.Linear(dim, dim),#将 emb 展平成一个形状为 (T, d_model) 的张量。
        )
        self.initialize()#初始化权重和偏置

    def initialize(self):
        for module in self.modules(): #遍历self.modules()中的所有模块
            if isinstance(module, nn.Linear):#检查模块是否是 nn.Linear 类型
                init.xavier_uniform_(module.weight)#使用 Xavier 均匀初始化方法初始化权重
                init.zeros_(module.bias)#初始化偏置为 0

    def forward(self, t):
        emb = self.timembedding(t)#将时间步 t 转换为时间嵌入向量
        return emb#返回时间嵌入向量


class DownSample(nn.Module):#下采样层
    """
    下采样层的主要功能是将输入特征图的空间分辨率降低一半。具体实现方式是:
    1. 使用3x3卷积核进行卷积操作
    2. 设置stride=2实现下采样,即每次滑动2个像素,这样输出特征图的高宽都会减半
    3. padding=1保证特征图边缘信息不会丢失
    4. 保持输入输出通道数相同(in_ch -> in_ch)
    5. 使用Xavier初始化方法初始化卷积层的权重,使用0初始化偏置

    这样可以在保留重要特征的同时减小特征图的空间维度,有助于:
    - 减少计算量和内存占用
    - 扩大感受野
    - 提取更高层次的特征

    关于高宽减半的原理:
    假设输入特征图大小为 H x W
    - padding=1 使特征图变为 (H+2) x (W+2)
    - 3x3卷积核配合stride=2,输出特征图大小计算公式:
      out_size = (in_size + 2*padding - kernel_size)/stride + 1
      height = (H + 2*1 - 3)/2 + 1 = H/2
      width = (W + 2*1 - 3)/2 + 1 = W/2
    因此输出特征图大小变为 (H/2) x (W/2)
    """
    def __init__(self, in_ch):#in_ch 是输入通道数
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)#使用 3x3 卷积核，步幅为 2，填充为 1
        self.initialize()#初始化权重和偏置

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        #使用 Xavier 均匀初始化方法初始化权重
        #保持网络层之间信号的方差大致恒定，有助于防止梯度消失或爆炸。
        #会从一个均匀分布 [-a, a] 中采样权重，其中 a 是根据该层的输入和输出神经元数量计算出来的
        init.zeros_(self.main.bias)#初始化偏置为 0

    def forward(self, x, temb):
        x = self.main(x)#使用 3x3 卷积核，步幅为 2，填充为 1
        return x


class UpSample(nn.Module):
    """
    上采样层的主要功能是将输入特征图的空间分辨率扩大一倍。具体实现方式是:
    1. 首先使用最近邻插值方法(nearest neighbor interpolation)将特征图的高和宽扩大2倍
    2. 然后使用3x3卷积核进行卷积操作,步幅为1,填充为1
    3. 保持输入输出通道数相同(in_ch -> in_ch)
    4. 使用Xavier初始化方法初始化卷积层的权重,使用0初始化偏置

    这样可以在扩大特征图空间维度的同时保持特征的连续性:
    - 最近邻插值是一种简单的上采样方法,通过复制像素值来扩大特征图
    - 后续的卷积操作可以帮助平滑特征,减少上采样造成的块状伪影
    - 这种结构常用于生成模型中,用于逐步恢复图像的空间细节

    关于尺寸变化:
    假设输入特征图大小为 H x W
    1. 经过scale_factor=2的插值后,特征图变为 (2H) x (2W)
    2. 3x3卷积(stride=1,padding=1)保持特征图大小不变
    因此最终输出特征图大小为 (2H) x (2W)
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)  # 使用3x3卷积核,步幅1,填充1
        self.initialize()  # 初始化权重和偏置

    def initialize(self):
        init.xavier_uniform_(self.main.weight)  # 使用Xavier均匀初始化方法初始化权重
        init.zeros_(self.main.bias)  # 初始化偏置为0

    def forward(self, x, temb):
        _, _, H, W = x.shape  # 获取输入特征图的高和宽
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')  # 使用最近邻插值将特征图扩大2倍
        x = self.main(x)  # 使用3x3卷积进行特征平滑
        return x


class AttnBlock(nn.Module):
    """
    自注意力模块,用于捕获图像中的长距离依赖关系
    实现了类似Transformer中的自注意力机制,但是是在2D特征图上进行的
    能够捕捉图像中不同空间位置之间的长距离依赖关系，而不仅仅是依赖于卷积层的局部感受野。这对于理解全局上下文和生成更相关的细节很有帮助。
    总的来说，这个自注意力块就是让图片里的每个像素点：
    产生三个身份：查询 Q、键 K、值 V。
    用自己的 Q 去和所有其他点的 K 比较，计算相关度（注意力权重）。
    根据算出来的权重，把所有其他点的 V (信息内容) 加权混合起来，得到自己的新特征。
    最后把这个新特征加回到原来的自己身上。
    这样，每个点都能有效地融合来自图片全局的信息，尤其是那些和它相关但可能离得很远的信息。
    Query (Q):
    代表:
    当前正在处理的某个元素（在这里是特征图上的一个像素位置）发出的 "查询"。
    作用:
    它用来与其他所有位置的 "键" (K) 进行比较，以找出哪些位置的信息与当前位置最相关。可以理解为，当前像素在问：“为了更新我自己的特征，我应该关注图像中的哪些其他部分？”
    Key (K):
    代表:
    所有元素（所有像素位置）提供的 "标识" 或 "键"。
    作用:
    它用来 被 "查询" (Q) 进行匹配。每个 Key 向量 k_j 相当于位置 j 在说：“这是我所包含信息的‘标签’，你可以用你的 Query 来和我匹配，看看我们有多相关。”
    Value (V):
    代表:
    所有元素（所有像素位置）实际包含的 "内容" 或 "值"。
    作用:
    一旦通过 Q 和 K 的交互计算出注意力权重（即确定了每个 Query 应该关注哪些 Key 的程度），这些权重就会被用来对相应的 Value 向量进行加权求和。
    Value 向量 v_j 包含了位置 j 要贡献给最终输出的实际特征信息
    """
    def __init__(self, in_ch):
        super().__init__()
        # 首先进行组归一化,将特征分成32组进行归一化,有助于训练稳定性
        self.group_norm = nn.GroupNorm(32, in_ch)
        # 使用1x1卷积生成query向量,保持通道数不变
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)#输入/输出/卷积核大小(为1相当于一个跨通道的全连接层，不改变空间维度)/步长（每次滑动的步长，为1个像素）/填充为0
        # 使用1x1卷积生成key向量
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # 使用1x1卷积生成value向量
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # 最后的投影层,用于将注意力的输出映射回原始通道数
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        # 对所有投影层使用Xavier初始化,使得前向传播时方差保持不变
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)#保持网络层之间信号的方差大致恒定，有助于防止梯度消失或爆炸。它会从一个均匀分布 [-a, a] 中采样权重，其中 a 是根据该层的输入和输出神经元数量计算出来的
            init.zeros_(module.bias)
        # 最后的投影层使用较小的增益以提高训练稳定性
        init.xavier_uniform_(self.proj.weight, gain=1e-5)
        #这个参数允许手动乘以上面计算出的标准边界 a。也就是说，实际的采样范围变成了 [-a * gain, a * gain]
        #这里gain的值非常小，这会导致 self.block2[-1] (第二个卷积块的最后一个卷积层) 的初始权重被采样自一个 非常接近于零 的范围 [-a * 1e-5, a * 1e-5]。换句话说，这一层的权重在初始化时几乎都是零。

    def forward(self, x):
        # 获取输入特征图的维度信息，批次大小，通道数，高度，宽度
        B, C, H, W = x.shape
        # 首先进行组归一化
        h = self.group_norm(x)
        # 生成query, key, value三个投影
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # 调整query的形状为(B, H*W, C),将空间维度展平，从 [B, C, H, W] 变成 [B, H*W, C]，q 就像是一个列表，每个批次 B 里有 N 个元素，每个元素是一个长度为 C 的查询向量。
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        # 调整key的形状为(B, C, H*W)
        k = k.view(B, C, H * W)
        # 计算注意力权重,看看谁和谁最相关，点积结果越大，表示 q_i 和 k_j 越相似或相关。
        # 并进行缩放以避免梯度消失
        w = torch.bmm(q, k) * (int(C) ** (-0.5))#缩放因子被选为点积计算中每个向量维度 d_k，即这里的C平方根的倒数，同时int（C）确保是一个整数
        # 验证注意力矩阵的形状正确
        assert list(w.shape) == [B, H * W, H * W]
        # 对注意力权重进行softmax归一化
        w = F.softmax(w, dim=-1)
        # 使用 Softmax 函数处理原始分数 w。
        #   - 对于每个查询点 i，它对所有其他点 j 的关注度分数会被转换成一个概率分布（所有分数加起来等于 1）。
        #   - 现在 w[b, i, j] 表示第 b 张图中，第 i 个点应该给第 j 个点的“关注度权重”（0到1之间）。
        #   - 关注度高的点，权重就大；关注度低的点，权重就小。

        # 调整value的形状为(B, H*W, C)，把 v 的形状也变成 [B, N, C]，和 q 一样。每个元素是长度为 C 的值向量。
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)

        # 将注意力权重与value相乘得到输出
        h = torch.bmm(w, v)
        # 计算加权和。用注意力权重矩阵 w (形状 [B, N, N]) 去乘以值矩阵 v (形状 [B, N, C])。
        #   - 对于每个输出位置 i，它的新特征 h[b, i] 是所有位置 j 的值向量 v[b, j] 的加权平均，权重就是 w[b, i, j]。
        #   - 简单说，每个点 i 的新特征，是根据它对其他所有点 j 的关注度，把这些点 j 的“信息内容”（v_j）按比例混合起来得到的。
        #   - 结果 h 的形状是 [B, N, C]。
        # 验证输出形状正确
        assert list(h.shape) == [B, H * W, C]

        # 将输出重新排列为2D特征图形状
        # 把加权聚合后的结果 h 从序列形状 [B, N, C] 变回图片特征图的形状 [B, C, H, W]。
        #   - `view(B, H, W, C)`: 先变回 [B, H, W, C]
        #   - `permute(0, 3, 1, 2)`: 再把通道 C 挪回第二个位置，变成 [B, C, H, W]。
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        # 通过最后的投影层
        h = self.proj(h)

        # 添加残差连接,将注意力的输出加到原始输入上
        return x + h
        # 最关键的一步：将注意力模块计算得到的输出 h，加回到原始的输入 x上。
        # 这叫做“残差连接”。
        #   - 好处是，模型既保留了原始信息 x，又获得了注意力模块提供的全局上下文信息 h。
        #   - 这样模型更容易训练，效果也通常更好。它让注意力模块专注于学习对原始特征的“补充”或“修正”。


class ResBlock(nn.Module):
    """
    残差块,用于构建UNet的残差连接
    每个残差块包含两个卷积层,一个用于特征提取,一个用于特征增强
    还包含一个用于时间步长嵌入的投影层
    最后通过残差连接将输入与输出相加
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):#in_ch是输入通道数，out_ch是输出通道数，tdim=ch*4是时间步长嵌入的维度，dropout是dropout率，attn是是否使用注意力模块
        # 继承父类初始化
        super().__init__()
        #主分支
        #第一个卷积块,包含归一化、激活和卷积层
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),#组归一化,将通道分为32组进行归一化
            Swish(),#Swish激活函数,相比ReLU有更好的性能，增加非线性
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),#最后通过一个 3x3 卷积层，可能改变通道数 (从 in_ch 到 out_ch)，stride=1, padding=1 保证卷积后特征图空间尺寸不变
        )
        #时间步长嵌入投影层,将时间信息注入到特征中
        self.temb_proj = nn.Sequential(
            Swish(),  #先对传入的时间嵌入 temb 应用 Swish 激活
            nn.Linear(tdim, out_ch),
            # 然后通过一个全连接层 (Linear)，将时间嵌入的维度 tdim 映射到目标输出通道数 out_ch
            # 这是为了让时间信息能和 block1 的输出特征图 h 相加
        )
        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), #对 block1 和时间信息融合后的结果进行组归一化 (注意通道数现在是 out_ch)
            Swish(),  # Swish 激活
            nn.Dropout(dropout),  # 应用 Dropout 防止过拟合
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1), #再通过一个 3x3 卷积，通道数保持 out_ch 不变
        )
        # 残差连接的shortcut,当输入输出通道不同时需要1x1卷积进行调整，shortcut分支
        if in_ch != out_ch:
        # 如果输入通道数 in_ch 和输出通道数 out_ch 不一样
        # 我们需要用一个 1x1 卷积来调整原始输入 x 的通道数，使其变为 out_ch
        # 这样才能和主分支的输出 h (通道数为 out_ch) 相加
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)#当输入输出通道不同时需要1x1卷积进行调整
        else:
            self.shortcut = nn.Identity()  # 通道数相同时直接使用恒等映射
        # 是否使用注意力模块
        if attn:
        # 如果 attn 参数为 True
        # 就在这个块的最后添加一个 AttnBlock (自注意力层)
        # 注意力层的输入输出通道数都是 out_ch
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()  # 不使用注意力时用恒等映射
        self.initialize()  # 初始化模型参数

    def initialize(self):
        """
        初始化模型参数的方法:
        1. 对所有卷积层和线性层进行Xavier初始化,这种初始化方法可以使得每一层的输出方差大致相等,
           有助于解决深度网络中的梯度消失和梯度爆炸问题
        2. 将所有偏置项初始化为0
        3. 对第二个卷积块的最后一层使用较小的增益(1e-5),这样可以降低残差分支的初始贡献,
           使得网络在训练初期更依赖shortcut分支,提高训练的稳定性
        """
        # 对所有卷积层和线性层进行Xavier初始化
        for module in self.modules():#遍历self.modules()中的所有模块
            if isinstance(module, (nn.Conv2d, nn.Linear)):#检查模块是否是nn.Conv2d或nn.Linear类型
                init.xavier_uniform_(module.weight)  # 权重使用Xavier初始化
                init.zeros_(module.bias)  # 偏置初始化为0
        # 第二个卷积块的最后一层使用较小的增益以提高训练稳定性
        # self.block2[-1] 指的是 self.block2 (Sequential) 里的最后一个层，即 Conv2d
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):#x是输入特征图，temb是时间步长嵌入向量
        #前向传播
        #主分支
        h = self.block1(x)  # 通过第一个卷积块
        h += self.temb_proj(temb)[:, :, None, None] #增加了两个维度
        # 时间嵌入向量 temb 通过时间投影层 self.temb_proj
        # 这个投影层包含 Swish 激活和一个线性层，将 temb 的维度从 tdim 映射到 out_ch，使其维度与 h 的通道数匹配
        # 将形状为 (batch_size, out_ch, 1, 1) 的时间嵌入加到形状为 (batch_size, out_ch, height, width) 的特征图 h 上。
        h = self.block2(h)  # 通过第二个卷积块，这个块包含 GroupNorm、Swish、Dropout 和 Conv2d。它对特征进行进一步的处理和细化。
        #残差连接，shortcut分支
        h = h + self.shortcut(x) #添加残差连接，缓解梯度消失（梯度在反向传播时变得很小，接近于零），x是最原始的输入特征图
        #如果 in_ch == out_ch，self.shortcut 是 nn.Identity()，直接返回 x 本身。
        #如果 in_ch != out_ch，self.shortcut 是一个 1x1 卷积层，它会将 x 的通道数从 in_ch 调整为 out_ch，使其与 h 的通道数匹配，同时保持空间维度不变。
        #注意力机制
        h = self.attn(h)  # 通过注意力层
        return h


class UNet(nn.Module):
    """
    UNet是扩散模型中的主要网络架构,用于学习噪声预测。主要特点包括:

    1. 网络结构:
    - 采用U型结构,包含下采样路径(encoder)和上采样路径(decoder)
    - 使用跳跃连接(skip connection)将对应层的特征图连接起来
    - 包含多个ResBlock块,可选的注意力机制

    2. 主要组件:
    - time_embedding: 时间步长编码层,将时间信息嵌入到网络中
    - head: 输入层,3通道图像转换为ch通道特征
    - downblocks: 下采样模块,逐步降低特征图分辨率
    - middleblocks: 中间层,在最低分辨率上处理特征
    - upblocks: 上采样模块,逐步恢复特征图分辨率
    - tail: 输出层,生成3通道的噪声预测

    3. 参数说明:
    - T: 总的时间步数
    - ch: 基础通道数
    - ch_mult: 各层通道数的倍数列表，通过控制 U-Net 不同分辨率层级的通道数量，使得网络能够在下采样时有效提取和压缩信息（增加通道，减少空间），
               并在上采样时结合高层语义和低层细节进行精确重建（减少通道，增加空间）。这是平衡网络表达能力、计算效率和捕捉多尺度特征的关键机制。
    - attn: 添加注意力机制的层索引
    - num_res_blocks: 每层的ResBlock数量
    - dropout: dropout比率
    """
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):#T是总的时间步数，ch是基础通道数，ch_mult是各层通道数的倍数列表，attn是添加注意力机制的层索引，num_res_blocks是每层的ResBlock数量，dropout是dropout比率
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]) #检查attn中的索引是否在ch_mult列表的范围内，attn[2]必须在ch_mult中
        tdim = ch * 4 #计算时间嵌入向量的目标维度，这里设置为基础通道数的 4 倍，empirical design choice经验设计
        self.time_embedding = TimeEmbedding(T, ch, tdim) #用于将时间步 t 转换为维度为 tdim 的嵌入向量

        # 输入卷积层
        # 这是一个 3x3 的卷积层。它接收输入的 3 通道图像（RGB），并将其转换为具有 ch 个通道的初始特征图。stride=1, padding=1 确保卷积后特征图的空间尺寸（高度和宽度）保持不变。
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # 下采样路径
        self.downblocks = nn.ModuleList()#创建一个ModuleList，用于存储下采样路径的所有层（ResBlock和DownSample）
        chs = [ch]  #记录各层输出通道数,用于上采样时的跳跃连接
        now_ch = ch #跟踪当前层的输入通道数
        for i, mult in enumerate(ch_mult):#遍历ch_mult列表中的每个元素
            out_ch = ch * mult#计算当前层的输出通道数
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(#添加ResBlock块，两块
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    #in_ch=now_ch: 输入通道是上一层的输出通道。
                    #out_ch=out_ch: 输出通道是当前层级计算的目标通道数。
                    #tdim=tdim: 传递时间嵌入维度。
                    #dropout=dropout: 传递 dropout 比率。
                    dropout=dropout, attn=(i in attn)))#判断当前层级索引 i 是否在 attn 列表中，如果是，则在此 ResBlock 内部启用注意力机制。
                now_ch = out_ch#更新now_ch 为刚刚添加的 ResBlock 的输出通道数，作为下一个块的输入通道数。
                chs.append(now_ch)#将每个 ResBlock 的输出通道数添加到 chs 列表。这是为了在上采样时精确匹配跳跃连接。
            if i != len(ch_mult) - 1:#检查是否是 ch_mult 中的最后一个层级。如果不是最后一个层级，则需要进行下采样。
                self.downblocks.append(DownSample(now_ch))#添加一个 DownSample 层（通常是步长为 2 的卷积或池化），它会将特征图的空间分辨率减半，通道数保持 now_ch 不变。
                chs.append(now_ch)#在下采样之后，再次将当前通道数 now_ch 添加到 chs 列表。这对应于下采样层的输出特征。

        # 中间层，在空间分辨率最低的地方，添加注意力机制，来捕捉全局上下文信息
        # 在瓶颈处应用注意力机制，其计算成本相对于在更高分辨率的层级应用要低得多
        # 这使得在不显著增加计算负担的情况下，也能获得注意力带来的好处
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),#添加第一个中间层 ResBlock。输入和输出通道数都是下采样路径最终的通道数 now_ch。强制启用注意力机制 (attn=True)。
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),#添加第二个中间层 ResBlock。输入输出通道数不变，不启用注意力机制 (attn=False)。
        ])

        # 上采样路径，特征图从低分辨率逐步恢复到高分辨率，同时通过跳跃连接(skip connection)融合编码器对应层的特征
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):#反向遍历ch_mult列表
            out_ch = ch * mult #计算当前上采样层级的目标输出通道数。
            for _ in range(num_res_blocks + 1):#在每个上采样层级，循环添加3个ResBlock块，多出来的一次是为了处理与跳跃连接拼接后的特征，1个用于特征融合，2个用于特征增强
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    #in_ch=chs.pop() + now_ch: 计算输入到 ResBlock 的通道数
                    #chs.pop(): 从 chs 列表末尾弹出一个通道数。由于 chs 记录了下采样路径各阶段的通道数，并且我们是反向遍历，pop() 会取出对应层级的下采样路径的特征图通道数（来自跳跃连接）
                    #now_ch: 上一层（或 UpSample 层）的输出通道数。
                    #两者相加，表示将跳跃连接的特征与上采样路径的特征在通道维度上拼接 (concatenate) 后的总通道数。
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch #更新 now_ch 为当前 ResBlock 的输出通道数
            if i != 0:#检查是否是第一个层级（对应原始分辨率）。如果不是，则需要进行上采样
                self.upblocks.append(UpSample(now_ch))#添加一个 UpSample层，它会将特征图的空间分辨率扩大一倍，通道数保持 now_ch 不变
        assert len(chs) == 0#在所有上采样层处理完毕后，chs 列表应该为空，表示所有来自下采样路径的跳跃连接特征都已经被 pop() 并使用了。

        # 输出层，将特征图转换为3通道图像
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),#组归一化，将特征图分成32组进行归一化
            Swish(),#Swish 激活函数，相比ReLU有更好的性能
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)# 最后一个 3x3 卷积层，将特征图的通道数从 now_ch 转换回 3（对应预测的噪声图像的 RGB 通道），保持空间分辨率不变。
        )
        self.initialize()#初始化网络参数

    def initialize(self):
        """初始化网络参数"""
        init.xavier_uniform_(self.head.weight)#使用 Xavier 均匀分布初始化 head 卷积层的权重。
        init.zeros_(self.head.bias)#将 head 卷积层的偏置初始化为 0。
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        #使用Xavier均匀初始化 方法初始化权重，gain=1e-5
        #gain=1e-5 的作用是使得 ResBlock 中残差计算路径的最后一个卷积层的初始权重非常小，接近于零。这有助于在训练初期稳定网络，让残差块的行为更接近于恒等映射
        #有助于稳定训练初期，让网络从一个接近零的预测开始学习。
        init.zeros_(self.tail[-1].bias)#将 tail 输出卷积层的偏置初始化为 0。

    def forward(self, x, t):#输入图像张量，形状为(batch_size, 3, 32, 32)，t是时间步长，形状为(batch_size,)
        """
        前向传播过程:
        1. 生成时间嵌入
        2. 下采样路径处理
        3. 中间层处理
        4. 上采样路径处理(包含跳跃连接)
        5. 输出最终预测
        """
        # 时间步长编码
        temb = self.time_embedding(t)#生将输入的时间步 t 通过 time_embedding 层，生成时间嵌入向量 temb。
        # 下采样过程
        h = self.head(x)#输入图像通过输入卷积层，生成初始特征图将3通道图像转换为128通道的特征图h
        hs = [h]#初始化列表 hs，用于存储跳跃连接所需的所有中间特征图。首先存入 head 的输出。
        for layer in self.downblocks:#遍历 downblocks 列表中的每一层（ResBlock 或 DownSample）
            h = layer(h, temb)#将当前特征图 h 和时间嵌入 temb 传递给当前层 layer。DownSample 层不需要 temb，但 ResBlock 需要。其内部会处理 temb
            hs.append(h)#记录每个下采样层的特征图,将当前层 layer 的输出 h 添加到 hs 列表中，保存下来用于后续的跳跃连接。
        # 中间层处理
        for layer in self.middleblocks:#遍历中间层
            h = layer(h, temb)#将当前特征图 h 和时间嵌入 temb 传递给中间层的 ResBlock,生成新的特征图
        # 上采样过程
        for layer in self.upblocks:#遍历上采样层
            if isinstance(layer, ResBlock):#检检查当前层是否是 ResBlock。跳跃连接发生在 ResBlock 的输入处。
                h = torch.cat([h, hs.pop()], dim=1)  # 跳跃连接，将特征图和跳跃连接的特征图进行拼接
                #hs.pop(): 从 hs 列表末尾弹出之前存储的、对应层级的下采样路径特征图。
                #torch.cat([h, ...], dim=1): 将当前上采样路径的特征图 h 与弹出的跳跃连接特征图，在通道维度 (dim=1) 上进行拼接 (concatenate)
            h = layer(h, temb)#将（可能经过拼接的）特征图 h 和时间嵌入 temb 传递给当前层 layer（ResBlock 或 UpSample）
        h = self.tail(h)#将特征图输入到输出层，生成最终的预测

        assert len(hs) == 0#检查hs列表是否为空，确保所有特征图都被处理，如果hs列表不为空，说明有特征图未被处理
        return h #返回最终预测的噪声图 h，其形状应与输入图像 x 相同 (batch_size, 3, height, width)

    #用于获取中间层的特征图，可以获取不同层次的特征图，用于分类任务
    def get_multiple_features(self, x, timesteps, block_num_lst=[24], return_num_params=False):
        bn=0 #块计数器，用于跟踪当前处理到第几个块，包括 downblocks, middleblocks, upblocks 里的每个 ResBlock 或 DownSample/UpSample 层）。
        return_lst = [] #初始化一个空列表 return_lst。这个列表将用来存储那些编号在 block_num_lst 中的块所输出的特征图。
        temb = self.time_embedding(timesteps)#生成时间嵌入向量，将时间步转换为高维表示
        #print(len(self.downblocks)) #11
        h = self.head(x) #输入图像通过输入卷积层，生成初始特征图将3通道图像转换为128通道的特征图
        hs = [h] #将初始特征图存入hs列表
        for ct, module in enumerate(self.downblocks):#遍历下采样层的ModuleList,enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列（ct），同时列出数据和数据下标
            h = module(h, temb)#将特征图和时间嵌入向量输入到模块中，生成新的特征图，得到新的特征图
            hs.append(h)#保存新的特征图,将当前模块的输出 h 添加到 hs 列表，供后续跳跃连接使用
            bn+=1#块计数器加1，记录当前处理到第几个块

            if bn in block_num_lst:#如果当前块的编号在block_num_lst列表中
                return_lst.append(h)#将该块的特征保存到返回列表中
        #print(hs,len(hs))
        # Middle
        for layer in self.middleblocks: #遍历中间层
            h = layer(h, temb)#将特征图和时间嵌入向量输入到模块中，生成新的特征图
            bn+=1#块计数器加1，记录当前处理到第几个块
            if bn in block_num_lst:#如果当前块的编号在block_num_lst列表中
                return_lst.append(h)
        # Upsampling
        for layer in self.upblocks: #遍历上采样层的ModuleList
            if isinstance(layer, ResBlock):#检查是否是ResBlock块
                h = torch.cat([h, hs.pop()], dim=1)
                #如果是 ResBlock，执行跳跃连接。从 hs 列表中弹出对应的下采样特征图，并与当前上采样特征图 h 在通道维度 (dim=1) 上拼接。
            h = layer(h, temb)#将特征图和时间嵌入向量输入到模块中，生成新的特征图
            bn+=1#块计数器加1，记录当前处理到第几个块
            if bn in block_num_lst:#如果当前块的编号在block_num_lst列表中
                return_lst.append(h)#将该块的特征保存到返回列表中
        #print(bn) #28
        #返回特征列表
        return return_lst


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)

