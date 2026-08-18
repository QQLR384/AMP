import torch
import torch.nn as nn

# ==========================================
# 4. ANIA Network Architecture 
# ==========================================
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class FCGRInceptionModule(nn.Module):
    def __init__(self, in_channels=11, out_channels=256, b1=64, b3=96, b3r=64, b5=64, b5r=48, bp=32):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, b1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, b3r, kernel_size=1),
            BasicConv2d(b3r, b3, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(b3, b3, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channels, b5r, kernel_size=1),
            BasicConv2d(b5r, b5r, kernel_size=3, padding=1),
            BasicConv2d(b5r, b5, kernel_size=3, padding=1)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, bp, kernel_size=1)
        )
        self.branch_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        bp = self.branch_pool(x)
        w = torch.softmax(self.branch_weights, dim=0)
        out = torch.cat([b1*w[0], b3*w[1], b5*w[2], bp*w[3]], dim=1)
        return out

class ANIA(nn.Module):
    def __init__(self, in_channels=11, d_model=512, dropout=0.3):
        super().__init__()
        self.inception1 = FCGRInceptionModule(in_channels=in_channels, out_channels=192, b1=48, b3=64, b3r=48, b5=48, b5r=32, bp=32)
        self.inception2 = FCGRInceptionModule(in_channels=192, out_channels=256, b1=64, b3=96, b3r=64, b5=64, b5r=48, bp=32)
        self.res_proj = nn.Conv2d(192, 256, kernel_size=1)
        self.spatial_pool = nn.MaxPool2d(2, 2)
        self.projection = nn.Sequential(nn.Linear(256, d_model//2), nn.ReLU(True), nn.Linear(d_model//2, d_model))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*2, dropout=dropout, batch_first=True), num_layers=2)
        self.dense = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.inception1(x)
        res = self.res_proj(x)
        x = self.inception2(x) + res
        x = self.spatial_pool(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.projection(x)
        x = self.transformer(x)
        return self.dense(x.mean(dim=1))