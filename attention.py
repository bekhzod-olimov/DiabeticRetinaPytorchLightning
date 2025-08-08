import torch

class HybridAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Spatial attention
        self.spatial_att = torch.nn.Sequential( torch.nn.Conv2d(in_channels, 1, kernel_size=1), torch.nn.Sigmoid() )
        # Self-attention
        self.self_att = torch.nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

    def forward(self, x):
        # Spatial attention
        att_map = self.spatial_att(x)
        x_masked = x * att_map
        
        # Self-attention
        b, c, h, w = x_masked.shape
        x_flat = x_masked.view(b, c, -1).permute(0, 2, 1)
        att_out, _ = self.self_att(x_flat, x_flat, x_flat)
        return att_out.permute(0, 2, 1).view(b, c, h, w)
    
class HybridAttention1D(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # Self-attention only (no spatial attention)
        self.self_att = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=4, batch_first=True)
    
    def forward(self, x):
        # x shape: [B, 512]
        x = x.unsqueeze(1)  # [B, 1, 512]
        att_out, _ = self.self_att(x, x, x)
        return att_out.squeeze(1)  # [B, 512]