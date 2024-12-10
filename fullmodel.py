import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers):
        super(TransformerModel, self).__init__()
        self.position_encoding = nn.Parameter(torch.randn(1, 512, d_model))  
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1) 
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)  
        return x
 

class Transformer(nn.Module):
    def __init__(self, n_channels, num_heads=4, att_drop=0.05, act='leaky_relu', input_size=64, hidden_size=64, num_layers=1):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        # 4 4
        self.query = nn.Linear(self.n_channels, self.n_channels//8)
        self.key   = nn.Linear(self.n_channels, self.n_channels//8)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.query1 = nn.Linear(self.n_channels, self.n_channels//4)
        self.key1 = nn.Linear(self.n_channels, self.n_channels//4)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        self.act = torch.nn.LeakyReLU(0.2)
        
        self.reset_parameters()
        
        self.fc1 = nn.Linear(n_channels*4, 32)
    
        self.conv_transformer = TransformerModel(d_model=32, nhead=4, num_encoder_layers=1)

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        one_tensor = x[:, 0, :4, :]
        two_tensor = x[:, 1, :, :]

        x = one_tensor
        y = two_tensor
        
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))
        H = 1
        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) 
        x = o.permute(0,2,1,3).reshape((B, M, C)) + x
        x = self.fc1(x.reshape((B, M*C)))
        B, M, C = y.size() # batchsize, num_metapaths, channels
        y = self.conv_transformer(y)
        y = y.reshape((B, 32))
        x = x + y
        return x
        
class SiameseTransformer(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseTransformer, self).__init__()
        self.embedding_net = embedding_net
    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
        #return output1
    def get_embedding(self, x):
        return self.embedding_net(x)

