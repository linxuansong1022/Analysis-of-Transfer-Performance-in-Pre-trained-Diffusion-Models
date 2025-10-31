import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import nn
from einops import rearrange

#swish激活函数定义
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):#T是总的时间部署，d_model表示时间嵌入的维度，必须是偶数，因为后续会拆分成正弦和余弦，dim是最终输出的时间嵌入向量纬度
        assert d_model % 2 == 0 #检查是否是偶数
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        #这里如果d_model=8，会生成一个张量 [0, 2, 4, 6]，之所以使用步长为 2，是因为后续会将生成的位置编码分为正弦和余弦两部分，这样可以确保每个位置都有对应的正弦和余弦分量。
        #将生成的张量中的每个元素除以 d_model，目的是对生成的位置索引进行归一化，将其缩放到 [0, 1) 的范围内，避免数值过大。
        #乘以 math.log(10000)，这一步是为了调整频率的尺度。在位置编码中，不同的频率可以表示不同的时间步信息，通过这种方式可以让模型学习到不同时间步之间的相对关系。
        emb = torch.exp(-emb)#对 emb 中的每个元素取负后再进行指数运算。这样做是为了将之前计算得到的频率值转换为合适的范围，使得后续的正弦和余弦函数能够在不同的频率上振荡，从而为不同的时间步提供不同的编码。
        pos = torch.arange(T).float()#0 到 T-1 的浮点数张量 pos，表示不同的时间步。
        emb = pos[:, None] * emb[None, :] #将 pos 和 emb 进行广播相乘
        assert list(emb.shape) == [T, d_model // 2] # 确保 emb 的形状为 [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1) # 将 emb 拆分成正弦和余弦两部分，并在最后一维堆叠
        assert list(emb.shape) == [T, d_model // 2, 2] # 确保 emb 的形状为 [T, d_model // 2, 2]
        emb = emb.view(T, d_model) # 将 emb 重新调整形状为 [T, d_model]
        ## 定义时间嵌入的神经网络层
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb), # 使用预训练的嵌入层,为不同的时间步生成对应的嵌入向量
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self): #初始化方法，用于初始化网络层的权重和偏置
        for module in self.modules():
            if isinstance(module, nn.Linear): #是否是线性层
                nn.init.xavier_uniform_(module.weight)#Xavier均匀分布来初始化权重，缓解梯度消失和梯度爆炸
                nn.init.zeros_(module.bias)#偏置初始化为0

    def forward(self, t):
        # Convert time tensor to float
        emb = self.timembedding(t)#传入一个时间步索引t，得到对应的时间嵌入向量
        return emb

class DownSample1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)#一维卷积，卷积核大小3，步长2，填充1，输出长度减半
        self.initialize()#卷积层参数初始化

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)#初始化权重
        nn.init.zeros_(self.main.bias)#偏置为0

    def forward(self, x, temb):
        x = self.main(x)#x是输入数据，通过这个卷积层
        return x

class UpSample1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode='linear')#使用线性插值，输入数据的长度扩大两倍，实现上采样
        x = self.main(x)#将上采样后的数据通过卷积层进行处理
        return x

class AttnBlock1D(nn.Module): #和二维卷积中的注意力层的区别就是将Conv2d改为Conv1d
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)#输入特征x映射为查询向量Q的卷积层
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)#key
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)#Value
        self.proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)#注意力机制输出的结果进行投影的卷积层，注意力机制转换为合适的维度
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:#遍历q k v投影卷积层以及最终输出的投影卷积层
            nn.init.xavier_uniform_(module.weight)#权重初始化
            nn.init.zeros_(module.bias)#偏置初始化为0
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)#加了一个比较小的增益因子，会使初始化的权重值更接近0，初期稳定训练过程

    def forward(self, x):
        # 提取输入张量 x 的形状信息
        # B 代表批量大小（batch size），即一次处理的样本数量
        # C 代表通道数（channels），表示每个样本的特征数量
        # L 代表序列长度（sequence length），即每个样本的序列长度
        B, C, L = x.shape

        # 对输入张量 x 进行组归一化操作
        # 组归一化（Group Normalization）是一种归一化技术，它将通道分成多个组，并在每个组内进行归一化
        # 这里使用 self.group_norm 对输入 x 进行归一化，得到归一化后的张量 h
        h = self.group_norm(x)

        # 通过投影卷积层 self.proj_q 将归一化后的张量 h 映射为查询（Query）向量 q
        # 一维卷积操作 self.proj_q 会对 h 进行特征变换，得到查询向量
        q = self.proj_q(h)

        # 通过投影卷积层 self.proj_k 将归一化后的张量 h 映射为键（Key）向量 k
        # 一维卷积操作 self.proj_k 会对 h 进行特征变换，得到键向量
        k = self.proj_k(h)

        # 通过投影卷积层 self.proj_v 将归一化后的张量 h 映射为值（Value）向量 v
        # 一维卷积操作 self.proj_v 会对 h 进行特征变换，得到值向量
        v = self.proj_v(h)

        # 对查询向量 q 进行维度重排和形状调整
        # permute(0, 2, 1) 将 q 的维度从 (B, C, L) 重排为 (B, L, C)
        # view(B, L, C) 确保 q 的形状为 (B, L, C)
        q = q.permute(0, 2, 1).view(B, L, C)

        # 对键向量 k 进行形状调整，确保其形状为 (B, C, L)
        k = k.view(B, C, L)

        # 计算查询向量 q 和键向量 k 的矩阵乘法，并进行缩放操作
        # torch.bmm(q, k) 对 q 和 k 进行批量矩阵乘法，得到注意力分数矩阵
        # (int(C) ** (-0.5)) 是缩放因子，用于防止点积过大
        # 最终得到的 w 是注意力分数矩阵，形状为 (B, L, L)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))

        # 断言检查注意力分数矩阵 w 的形状是否为 (B, L, L)
        # 如果形状不符合预期，会抛出 AssertionError 异常
        assert list(w.shape) == [B, L, L]

        # 对注意力分数矩阵 w 进行 softmax 操作，得到注意力权重矩阵
        # F.softmax(w, dim=-1) 在最后一个维度上进行 softmax 操作，使得每行的元素之和为 1
        # 注意力权重矩阵 w 表示每个位置对其他位置的注意力程度
        w = F.softmax(w, dim=-1)

        # 对值向量 v 进行维度重排和形状调整
        # permute(0, 2, 1) 将 v 的维度从 (B, C, L) 重排为 (B, L, C)
        # view(B, L, C) 确保 v 的形状为 (B, L, C)
        v = v.permute(0, 2, 1).view(B, L, C)

        # 计算注意力权重矩阵 w 和值向量 v 的矩阵乘法，得到注意力输出 h
        # torch.bmm(w, v) 对 w 和 v 进行批量矩阵乘法，得到注意力输出
        # 注意力输出 h 的形状为 (B, L, C)
        h = torch.bmm(w, v)

        # 断言检查注意力输出 h 的形状是否为 (B, L, C)
        # 如果形状不符合预期，会抛出 AssertionError 异常
        assert list(h.shape) == [B, L, C]

        # 对注意力输出 h 进行维度重排和形状调整
        # view(B, L, C) 确保 h 的形状为 (B, L, C)
        # permute(0, 2, 1) 将 h 的维度从 (B, L, C) 重排为 (B, C, L)
        h = h.view(B, L, C).permute(0, 2, 1)

        # 通过投影卷积层 self.proj 对注意力输出 h 进行投影操作
        # 一维卷积操作 self.proj 会对 h 进行特征变换，得到最终的注意力特征
        h = self.proj(h)

        # 将输入张量 x 和注意力特征 h 进行残差连接
        # 残差连接可以帮助缓解梯度消失问题，提高模型的训练效果
        # 最终返回残差连接后的结果
        return x + h

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):#输入特征通道数，输出特征通道数，tdim是时间嵌入向量的维度，dropout是丢弃率
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),#组归一化
            Swish(),
            nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1),#卷积核大小3，步长1，填充1，输入输出序列长度不变
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),#将输入的时间嵌入向量从tdim维度投影到out_ch维度
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        #快捷链接shortcut
        if in_ch != out_ch:#如果输入通道数不等于输出通道数，则用一个1x1的卷积层来调整通道数
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0)
        else:#如果输入输出通道数相同，不进行操作，快捷链接
            self.shortcut = nn.Identity()
        if attn:#选择性插入注意力模块
            self.attn = AttnBlock1D(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):#如果模块是一维卷积层或者全连接层
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)#对最后一个卷积层的权重使用Xavier均匀分布初始化，并设置增益为1e-5

    def forward(self, x, temb):#输入特征x，和时间嵌入式向量temb
        h = self.block1(x)#将输入特征x通过第一个卷积块，得到中间特征图
        h += self.temb_proj(temb)[:, :, None]#将时间嵌入向量temb通过temb_proj进行投影，并在最后一个维度上添加一个维度，然后加到中间特征h上，实现时间信息的融合
        h = self.block2(h)#第二个卷积块
        h = h + self.shortcut(x)#将处理后的特征 h 与快捷连接的输出相加，实现残差连接，有助于缓解梯度消失问题，提高模型的训练效果
        h = self.attn(h)#如果 attn 为 True，则将特征 h 通过注意力模块 AttnBlock1D 进行处理；否则，直接返回 h。
        return h

class UNet1D(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):#总时间，基础通道数，通道倍增列表，控制哪些层用自注意力，每层残差块数（2），对丢弃率
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4 #计算时间嵌入向量的维度，为基础通道数的 4 倍
        self.time_embedding = TimeEmbedding(T, ch, tdim) #创建一个 TimeEmbedding 模块，用于将时间步转换为时间嵌入向量

        # 简化head层，直接使用1D卷积
        self.head = nn.Conv1d(1, ch, 3, stride=1, padding=1)

        # 下采样块
        self.downblocks = nn.ModuleList()
        chs = [ch] #初始化一个列表 chs 用于记录每个块的输出通道数，初始值为基础通道数 ch
        now_ch = ch #初始化当前通道数为基础通道数 ch
        for i, mult in enumerate(ch_mult): #遍历 ch_mult 列表，根据倍数计算每个分辨率下的输出通道数。
            out_ch = ch * mult #计算当前分辨率下的输出通道数
            for _ in range(num_res_blocks): #在每个分辨率下添加 num_res_blocks（2）个残差块。
                self.downblocks.append(ResBlock1D(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))#添加残差块
                now_ch = out_ch #更新当前通道数为 out_ch
                chs.append(now_ch) #将当前通道数添加到 chs 列表中
            if i != len(ch_mult) - 1: #如果不是最后一个分辨率，则添加一个下采样块
                self.downblocks.append(DownSample1D(now_ch)) #添加一个 DownSample1D 下采样块，输入通道数为 now_ch
                chs.append(now_ch) #将当前通道数添加到 chs 列表中

        #存储中间块，包括两个残差，第一个用注意力机制
        self.middleblocks = nn.ModuleList([
            ResBlock1D(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock1D(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        #上采样
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):#反向遍历 ch_mult 列表，进行上采样操作
            out_ch = ch * mult#计算当前分辨率下的输出通道数
            for _ in range(num_res_blocks + 1):#在每个分辨率下添加 num_res_blocks + 1 个残差块
                self.upblocks.append(ResBlock1D(#添加残差块
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))#输入通道数为 chs.pop() + now_ch（将下采样过程中记录的通道数取出并与当前通道数拼接）
                now_ch = out_ch#更新当前通道数为 out_ch
            if i != 0: #如果不是第一个分辨率，则添加一个上采样块
                self.upblocks.append(UpSample1D(now_ch))#添加一个 UpSample1D 上采样块，输入通道数为 now_ch。
        assert len(chs) == 0#确保 chs 列表中的元素全部被使用
        #尾部层，顺序容器将组归一化、Swish 激活函数和一维卷积层组合在一起，将输出通道数从 now_ch 减少到 1
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv1d(now_ch, 1, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)#初始化头部的权重
        nn.init.zeros_(self.head.bias)#头部的偏置初始化为0
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)#使用 Xavier 均匀分布初始化尾部层最后一个卷积层的权重，并设置增益为 1e-5
        nn.init.zeros_(self.tail[-1].bias)#将尾部层最后一个卷积层的偏置初始化为零

    def forward(self, x, t):#定义模型的前向传播方法，接收输入数据 x 和时间步 t。
        # 简化forward过程，直接处理3D输入
        temb = self.time_embedding(t)#将时间步 t 转换为时间嵌入向量 temb。
        h = self.head(x)  # 直接处理 [batch, 1, 128] 输入

        # 后续处理保持不变
        hs = [h]#初始化一个列表 hs 用于记录下采样过程中的特征
        for layer in self.downblocks:#遍历下采样块，对特征h进行下采样处理，并将每个块的输出特征添加到hs列表中
            h = layer(h, temb)
            hs.append(h)
        for layer in self.middleblocks:#遍历中间块，对特征 h 进行处理
            h = layer(h, temb)
        for layer in self.upblocks:#遍历上采样块，对特征 h 进行上采样处理
            if isinstance(layer, ResBlock1D): #如果当前层是残差块，则将特征 h 与下采样过程中记录的特征进行拼接，残差链接
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)#将特征 h 通过尾部层进行处理，得到最终输出
        return h

    def get_multiple_features(self, x, timesteps, block_num_lst=[16], return_num_params=False):#从模型的不同块中提取特征
        bn=0 #初始化一个变量 bn，用于记录当前处理的块的编号，初始值为 0
        return_lst = [] #初始化一个空列表 return_lst，用于存储提取的特征
        temb = self.time_embedding(timesteps) #调用 time_embedding 方法，将输入的 timesteps 转换为时间嵌入向量 temb
        #print(len(self.downblocks)) #11
        h = self.head(x)#head层
        hs = [h] #初始化一个列表 hs，用于存储下采样过程中的特征，初始值为头部层的输出h
        for ct, module in enumerate(self.downblocks): #遍历下采样块 self.downblocks，ct 是当前块的索引，module 是当前块的模块
            h = module(h, temb) #将特征 h 和时间嵌入向量 temb 传入当前下采样块 module，得到更新后的特征 h
            hs.append(h)#将更新后的特征 h 添加到列表 hs 中
            bn+=1 #块编号 bn 加 1

            if bn in block_num_lst: #检查当前块编号 bn 是否在 block_num_lst 列表中
                return_lst.append(h) #如果 bn 在 block_num_lst 中，则将当前特征 h 添加到 return_lst 列表中
        #print(hs,len(hs))
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
            bn+=1
            if bn in block_num_lst:
                return_lst.append(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock1D):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
            bn+=1
            if bn in block_num_lst:
                return_lst.append(h)
        #print(bn) #28
        #bb
        return return_lst
def get_multiple_features(self, x, timesteps, block_num_lst=[16], return_num_params=False):
    # 从模型的不同块中提取特征
    # 参数说明：
    # x: 输入数据，通常是一个张量
    # timesteps: 时间步，用于生成时间嵌入向量
    # block_num_lst: 一个列表，包含需要提取特征的块的编号，默认为 [16]
    # return_num_params: 是否返回模型的参数数量，默认为 False

    bn = 0  # 初始化一个变量 bn，用于记录当前处理的块的编号，初始值为 0
    return_lst = []  # 初始化一个空列表 return_lst，用于存储提取的特征
    temb = self.time_embedding(timesteps)  # 调用 time_embedding 方法，将输入的 timesteps 转换为时间嵌入向量 temb
    # print(len(self.downblocks))  # 11，这里注释掉的代码可以用于调试，查看下采样块的数量

    h = self.head(x)  # 将输入数据 x 通过模型的头部层（通常是一个卷积层），得到特征 h
    hs = [h]  # 初始化一个列表 hs，用于存储下采样过程中的特征，初始值为头部层的输出 h

    # 下采样阶段
    for ct, module in enumerate(self.downblocks):
        # 遍历下采样块 self.downblocks
        # ct 是当前块的索引，module 是当前块的模块（可以是残差块或下采样块）
        h = module(h, temb)  # 将特征 h 和时间嵌入向量 temb 传入当前下采样块 module，得到更新后的特征 h
        hs.append(h)  # 将更新后的特征 h 添加到列表 hs 中
        bn += 1  # 块编号 bn 加 1

        if bn in block_num_lst:  # 检查当前块编号 bn 是否在 block_num_lst 列表中
            return_lst.append(h)  # 如果 bn 在 block_num_lst 中，则将当前特征 h 添加到 return_lst 列表中

    # 中间块阶段
    for layer in self.middleblocks:
        # 遍历中间块 self.middleblocks
        h = layer(h, temb)  # 将特征 h 和时间嵌入向量 temb 传入当前中间块 layer，得到更新后的特征 h
        bn += 1  # 块编号 bn 加 1
        if bn in block_num_lst:  # 检查当前块编号 bn 是否在 block_num_lst 列表中
            return_lst.append(h)  # 如果 bn 在 block_num_lst 中，则将当前特征 h 添加到 return_lst 列表中

    # 上采样阶段
    for layer in self.upblocks:
        # 遍历上采样块 self.upblocks
        if isinstance(layer, ResBlock1D):  # 如果当前层是残差块
            h = torch.cat([h, hs.pop()], dim=1)  # 将特征 h 与下采样过程中记录的特征进行拼接，dim=1 表示在通道维度上拼接
        h = layer(h, temb)  # 将特征 h 和时间嵌入向量 temb 传入当前上采样块 layer，得到更新后的特征 h
        bn += 1  # 块编号 bn 加 1
        if bn in block_num_lst:  # 检查当前块编号 bn 是否在 block_num_lst 列表中
            return_lst.append(h)  # 如果 bn 在 block_num_lst 中，则将当前特征 h 添加到 return_lst 列表中

    # print(bn)  # 28，这里注释掉的代码可以用于调试，查看处理完所有块后的块编号
    # bb，这里可能是一个未完成的调试标记

    return return_lst  # 返回存储提取特征的列表 return_lst
if __name__ == '__main__':
    # Test the model
    batch_size = 8
    model = UNet1D(
        T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 1, 128)  # 1D input with length 128
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
    print(y.shape)  # Should output: torch.Size([8, 1, 128])
