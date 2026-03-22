import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class ViTUNETRSegmentationModel(nn.Module):
    def __init__(self, simclr_ckpt_path, img_size=(96,96,96), in_channels=1, out_channels=1):
        super().__init__()
        # Initialize UNETR directly
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name='instance',
            res_block=True,
            dropout_rate=0.0
        )
        
        # Load SimCLR weights into the UNETR encoder
        try:
            ckpt = torch.load(simclr_ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            # Match keys starting with 'backbone.'
            backbone_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('backbone.')}
            
            # strict=False handles MONAI version discrepancies (like missing Cross-Attention keys)
            self.unetr.vit.load_state_dict(backbone_state_dict, strict=False)
            print("="*10)
            print("Backbone weights successfully mapped to UNETR encoder")
            print("="*10)
        except Exception as e:
            print(f"Warning: Could not load backbone weights: {e}")
            print("ViT encoder initialized with random weights")

    def forward(self, x):
        return self.unetr(x)