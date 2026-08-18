import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 4. MBC-Attention Architecture
# ==========================================
class AttentionLocal1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, kernel_size=1)
        self.key = nn.Conv1d(channels, channels, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, L = x.shape
        q = self.query(x).transpose(1, 2)  
        k = self.key(x)                    
        v = self.value(x).transpose(1, 2)  
        
        attn_scores = torch.bmm(q, k) / math.sqrt(C) 
        local_weights = F.softmax(attn_scores, dim=-1) 
        weighted_v = torch.bmm(local_weights, v).transpose(1, 2) 
        return x + weighted_v

class AttentionGlobal(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(64)
        
    def forward(self, x):
        x_norm = self.ln(x)
        attn_scores = torch.bmm(x_norm, x_norm.transpose(1, 2)) / math.sqrt(64.0)
        global_weights = F.softmax(attn_scores, dim=-1)
        weighted_x = torch.bmm(global_weights, x) 
        return x + weighted_x

class CNNBranch1D(nn.Module):
    def __init__(self, input_dim, filters=32, dropout=0.4):
        super().__init__()
        # Ultimate fail-safe layer: Equivalent to the original CNNstandardInputOutput
        # Forces any abnormally scaled features back to a N(0,1) distribution before entering the network.
        self.input_norm = nn.BatchNorm1d(1, affine=False) 
        
        self.conv = nn.Conv1d(1, filters, kernel_size=3, padding=1)
        self.attn_local = AttentionLocal1D(filters)
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.flatten = nn.Flatten()
        
        pooled_size = math.ceil(input_dim / 2.0)
        flattened_dim = filters * pooled_size
        
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, 64),
            nn.LayerNorm(64), 
            nn.GELU()         
        )

    def forward(self, x):
        x = self.input_norm(x) # Feature dehydration, neutralizes astronomical scaling
        x = self.conv(x)
        x = F.gelu(x)         
        x = self.attn_local(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.dense(x)

class MBC_Attention(nn.Module):
    def __init__(self, p_dims):
        super().__init__()
        self.k_features = len(p_dims)
        self.branches = nn.ModuleList([
            CNNBranch1D(input_dim=p) for p in p_dims
        ])
        self.attn_global = AttentionGlobal()
        self.final_flatten = nn.Flatten()
        
        self.regressor = nn.Sequential(
            nn.Linear(64 * self.k_features, 64),
            nn.LayerNorm(64), 
            nn.GELU(),        
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, features_list):
        branch_outs = []
        for i in range(self.k_features):
            x = features_list[i].unsqueeze(1)
            out = self.branches[i](x)
            branch_outs.append(out.unsqueeze(1))
        
        concat_out = torch.cat(branch_outs, dim=1) 
        global_out = self.attn_global(concat_out)
        flat_out = self.final_flatten(global_out)
        return self.regressor(flat_out).squeeze(-1)