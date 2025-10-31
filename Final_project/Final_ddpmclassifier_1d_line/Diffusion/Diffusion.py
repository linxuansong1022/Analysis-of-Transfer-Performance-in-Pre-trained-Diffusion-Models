
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
    
    def __init__(self, layer_string): 
        super(AttentionHead, self).__init__()
        self.layer_string = layer_string
        self.layers = self.build_model()

        # Build CLS token based on layer string 
        self.use_cls_token, dim = self.parse_use_cls_token(layer_string.split(',')[0].strip())
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def build_model(self):
        layers = []
        layer_definitions = self.layer_string.split(',')

        for layer_definition in layer_definitions:
            layer_definition = layer_definition.strip()
            if layer_definition.startswith('Attention'):
                dim, num_heads, mlp_ratio, num_blocks = self.parse_block_layer(layer_definition)
                for i in range(num_blocks):
                    layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))
            elif layer_definition.startswith('Flatten'):
                layers.append(nn.Flatten())
            elif layer_definition.startswith('Dense'):
                in_dim, out_dim, activation = self.parse_dense_layer(layer_definition)
                layers.append(nn.Linear(in_dim, out_dim))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'identity':
                    pass
            elif layer_definition.startswith('Pool'):
                pool_size, stride = self.parse_pool_layer(layer_definition)
                layers.append(nn.AvgPool2d(pool_size, stride=stride))
            elif layer_definition.startswith('FullyPool'):
                layers.append(LambdaLayer(lambda x: x.mean(dim=1)))
            elif layer_definition.startswith('Use_CLS_Token'): 
                pass 
            elif layer_definition.startswith('Reshape'):
                layers.append(LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')))
            elif layer_definition.startswith('Insert_CLS_Token'):
                layers.append(LambdaLayer(lambda x: torch.cat((self.cls_token.to(x.device).expand(x.shape[0], -1, -1),
                                                               x), dim=1)))
            elif layer_definition.startswith('Extract_CLS_Token'):
                layers.append(LambdaLayer(lambda x: x[:, 0]))

        return nn.Sequential(*layers)

    def parse_use_cls_token(self, layer_definition):
        layer_definition = layer_definition.split(':')
        use_cls_token = bool(layer_definition[1])
        dim = int(layer_definition[2])
        return use_cls_token, dim
    
    def parse_pool_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')
        pool_size = int(layer_definition[1])
        stride = int(layer_definition[2])
        return pool_size, stride
    
    def parse_block_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')
        dim = int(layer_definition[1])
        num_heads = int(layer_definition[2])
        mlp_ratio = float(layer_definition[3])
        num_blocks = int(layer_definition[4])
        return dim, num_heads, mlp_ratio, num_blocks
    
    def parse_dense_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')
        in_dim = int(layer_definition[1])
        out_dim = int(layer_definition[2])
        activation = layer_definition[3]
        return in_dim, out_dim, activation


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() # I just default set this to identity function
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionClassifyTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, head, classify_T):
        super().__init__()
        self.head = head  
        self.classify_T = classify_T
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        if self.head=='fc':
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(8192, 10) #72+

        elif self.head=='mlp':
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Sequential(
                        nn.Linear(256*4*4, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, 10)
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
                        nn.Conv2d(256, 64, 3, 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 10, 3, 2),
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

        features = self.model.get_multiple_features(x_t, t)
        #print(len(features),features[0].shape)# [80, 256, 16, 16]
        
        if self.head=='fc':
            feature = self.pool(features[0])#.squeeze(-1).squeeze(-1)
            feature = torch.flatten(feature, start_dim=1)
            pre = self.fc(feature)

        elif self.head=='mlp':
            feature = self.pool(features[0])#.squeeze(-1).squeeze(-1)
            feature = torch.flatten(feature, start_dim=1)
            pre = self.fc(feature)

        elif self.head=='attn':
            feature = self.rer(features[0])
            
            # print(feature.shape) [80, 256, 1, 1]
            # feature = torch.flatten(feature, start_dim=1)
            # print(feature.shape)# [32, 65536]
            pre = self.fc(feature)
            pre = pre.mean(dim=1).squeeze(1)
    
        ### 4.CNN
        elif self.head=='cnn':
            feature = self.fc(features[0])
            pre = feature.mean(dim=3).mean(dim=2)
        loss = self.loss_fn( pre, labels)

        return pre,loss

