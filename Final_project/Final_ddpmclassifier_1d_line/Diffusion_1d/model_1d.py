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
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        # Convert time tensor to float
        emb = self.timembedding(t)
        return emb

class DownSample1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
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
        x = F.interpolate(x, scale_factor=2, mode='linear')
        x = self.main(x)
        return x

class AttnBlock1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, L = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 1).view(B, L, C)
        k = k.view(B, C, L)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, L, L]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 1).view(B, L, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, L, C]
        h = h.view(B, L, C).permute(0, 2, 1)
        h = self.proj(h)

        return x + h

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock1D(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet1D(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # 简化head层，直接使用1D卷积
        self.head = nn.Conv1d(1, ch, 3, stride=1, padding=1)
        
        # 其他层保持不变
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock1D(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample1D(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock1D(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock1D(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock1D(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample1D(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv1d(now_ch, 1, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # 简化forward过程，直接处理3D输入
        temb = self.time_embedding(t)
        h = self.head(x)  # 直接处理 [batch, 1, 128] 输入
        
        # 后续处理保持不变
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock1D):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        return h

    def get_multiple_features(self, x, timesteps, block_num_lst=[24], return_num_params=False):
        bn=0
        return_lst = []
        temb = self.time_embedding(timesteps)
        #print(len(self.downblocks)) #11
        h = self.head(x)
        hs = [h]
        for ct, module in enumerate(self.downblocks):
            h = module(h, temb)
            hs.append(h)
            bn+=1

            if bn in block_num_lst:
                return_lst.append(h)
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
