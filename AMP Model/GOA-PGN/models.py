import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 3. Loss Functions
# ==========================================
class WeightedMSELoss(nn.Module):
    def __init__(self, hp_threshold=1.0, hp_weight=5.0):
        super().__init__()
        self.hp_threshold = hp_threshold
        self.hp_weight = hp_weight
        
    def forward(self, pred, target):
        sq_err = (pred - target) ** 2
        weights = torch.where(target < self.hp_threshold, self.hp_weight, 1.0)
        return torch.mean(sq_err * weights)

# ==========================================
# 4. Architecture: Zero-Parameter Physical Graph + 1D Propagation
# ==========================================
class PConv1d(nn.Module):
    def __init__(self, dim=256, cp_dim=64, kernel_size=3):
        super().__init__()
        self.cp_dim = cp_dim
        self.untouched_dim = dim - cp_dim
        self.conv = nn.Conv1d(cp_dim, cp_dim, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(cp_dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.cp_dim, self.untouched_dim], dim=1)
        x1 = F.gelu(self.bn(self.conv(x1)))
        return torch.cat((x1, x2), dim=1)

class TaskSpecificHead(nn.Module):
    def __init__(self, in_channels, inner_dim, out_dim=1, reduction=8, is_classification=False):
        super().__init__()
        self.is_classification = is_classification
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, out_dim)
        )

    def forward(self, x):
        ca_weights = self.channel_attention(x)
        x_adapted = x * ca_weights
        out = self.regressor(x_adapted)
        return torch.sigmoid(out) if self.is_classification else out

class ZeroParamPhysGraph(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h_1d, coords, vec_sidechain, mask):
        # Keep the distance calculation exactly as is
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)
        
        v_i = F.normalize(vec_sidechain.unsqueeze(2), dim=-1)
        v_j = F.normalize(vec_sidechain.unsqueeze(1), dim=-1)
        cos_sim = torch.sum(v_i * v_j, dim=-1)
        
        adj = torch.exp(-dist / 3.0) * F.relu(cos_sim)
        
        mask_bool = mask.bool()
        mask_2d = mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)
        eye = torch.eye(dist.size(1), device=dist.device).unsqueeze(0).bool()
        
        adj = adj.masked_fill(~mask_2d | eye, 0.0)
        
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / deg
        
        h_phys = torch.matmul(adj_norm, h_1d)
        h_phys = F.gelu(self.proj(h_phys))
        
        return self.norm(h_1d + self.gamma * h_phys)

class DIR_ZeroParamPhys_Adapter(nn.Module):
    def __init__(self, input_dim=480, bottleneck_dim=256, hidden_dim=256):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.GELU(), nn.Dropout(0.2))
        self.pconv = PConv1d(dim=bottleneck_dim, cp_dim=64, kernel_size=3)
        self.phys_graph = ZeroParamPhysGraph(dim=bottleneck_dim)
        self.attention = nn.Sequential(nn.Linear(bottleneck_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.feature_extractor = nn.Sequential(nn.Linear(bottleneck_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
        
        self.mic_reg_head = TaskSpecificHead(in_channels=hidden_dim, inner_dim=64)
        self.mic_cls_head = TaskSpecificHead(in_channels=hidden_dim, inner_dim=64, is_classification=True)
        self.charge_head = TaskSpecificHead(in_channels=hidden_dim, inner_dim=32)
        self.gravy_head = TaskSpecificHead(in_channels=hidden_dim, inner_dim=32)

    def forward(self, emb, coords, vec, dih, mask):
        proj_emb = self.projector(emb)
        out_1d = self.pconv(proj_emb.transpose(1, 2)).transpose(1, 2)
        out_phys = self.phys_graph(out_1d, coords, vec, mask)
        
        attn_scores = self.attention(out_phys).masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled_feat = torch.sum(out_phys * attn_weights, dim=1)
        shared_features = self.feature_extractor(pooled_feat)
        
        return self.mic_reg_head(shared_features), self.mic_cls_head(shared_features), self.charge_head(shared_features), self.gravy_head(shared_features)