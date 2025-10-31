import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .model_1d import UNet1D

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, bias=qkv_bias)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (B, L, D)
        x = x + self._attention_block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
    def _attention_block(self, x):
        # x shape: (B, L, D) -> (L, B, D)
        x = x.transpose(0, 1)
        x, _ = self.attn(x, x, x)
        # x shape: (L, B, D) -> (B, L, D)
        return x.transpose(0, 1)

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

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer1D(nn.Module):
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
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        # 直接使用3D张量，无需维度调整
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
            
        predicted_noise = self.model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        return loss

class GaussianDiffusionTrainer1DReg(nn.Module):
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
        self.fc = nn.Linear(128, 1) #72+

    def forward(self, x_0, heights):
        #print(x_0.shape, heights.shape) 1280=bs*128
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        # 直接使用3D张量，无需维度调整
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
            
        features = self.model.get_multiple_features(x_t, t)
        #print(len(features),features[0].shape)# [1280, 128, 64]
        features = features[0].mean(2)
        #print(features.shape)
        pre = self.fc(features)
        heights = heights.unsqueeze(1) 
        loss = F.mse_loss(pre, heights, reduction='none')
        return loss


class GaussianDiffusionSampler1D(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coef1', torch.sqrt(1. / alphas))
        self.register_buffer('coef2', self.coef1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coef1, t, x_t.shape) * x_t -
            extract(self.coef2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

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
                device=x_0.device  # 保持与原张量相同的设备（GPU）
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


class GaussianDiffusionClassifyTrainer1d(nn.Module):
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
            self.pool = nn.AdaptiveAvgPool1d(4)
            self.fc = nn.Linear(256*4, 10)  # Adjust dimensions for 1D

        elif self.head=='mlp':
            self.pool = nn.AdaptiveAvgPool1d(4)
            self.fc = nn.Sequential(
                        nn.Linear(256*4, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, 10)
                    )

        elif self.head=='attn':
            self.rer = LambdaLayer(lambda x: rearrange(x, 'b c l -> b l c'))
            num_blocks = 1
            layer_string = f"Attention:256:8:4:{num_blocks},"
            self.fc = AttentionHead(layer_string)

        elif self.head=='cnn':
            self.fc = nn.Sequential(
                        nn.Conv1d(256, 64, 3, 2),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 10, 3, 2),
                    )

    def forward(self, x_0, labels):
        t = torch.full(
                size=(x_0.shape[0],), 
                fill_value=self.classify_T, 
                dtype=torch.long,
                device=x_0.device
            )
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

        features = self.model.get_multiple_features(x_t, t)
        
        if self.head=='fc':
            feature = self.pool(features[0])
            feature = torch.flatten(feature, start_dim=1)
            pre = self.fc(feature)

        elif self.head=='mlp':
            feature = self.pool(features[0])
            feature = torch.flatten(feature, start_dim=1)
            pre = self.fc(feature)

        elif self.head=='attn':
            feature = self.rer(features[0])
            pre = self.fc(feature)
            pre = pre.mean(dim=1).squeeze(1)

        elif self.head=='cnn':
            feature = self.fc(features[0])
            pre = feature.mean(dim=2)

        loss = self.loss_fn(pre, labels)
        return pre, loss


    
