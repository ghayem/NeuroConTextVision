import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from glob import glob
import os
import torch
import torch.nn as nn
from monai.networks.nets import ViT
import torch.nn.functional as F
import yaml
import pandas as pd

class ViTBackboneNet(nn.Module):
    def __init__(self, simclr_ckpt_path):
        super(ViTBackboneNet, self).__init__()
        
        # Create ViT backbone with same architecture as SimCLR
        self.backbone = ViT(
            in_channels=1,  # For single channel input
            img_size=(96,96,96),  # Adjust this to your input dimensions
            patch_size=(16, 16, 16),
            hidden_size=768,  # Standard for ViT-B
            mlp_dim=3072,
            num_layers=12,
            num_heads=12, 
            save_attn=True,
        )
        
        # Load pretrained weights from SimCLR checkpoint
        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Extract only backbone weights from SimCLR checkpoint
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                # Remove "backbone." prefix
                new_key = key[9:]  # len("backbone.") = 9
                backbone_state_dict[new_key] = value
        
        # Load the backbone weights
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        print("Backbone weights loaded!!")

    def forward(self, x):
        # Get features from ViT backbone
        features = self.backbone(x)
        
        # Use CLS token (first token) as global representation
        # features[0] shape: [batch_size, num_tokens, hidden_dim]
        # features[0][:, 0] gets CLS token: [batch_size, hidden_dim]
        cls_token = features[0][:, 0]  # Shape: [batch_size, 768]
        
        return cls_token

class Classifier(nn.Module):
    def __init__(self, d_model=768, num_classes=1):  # d_model=768 for ViT-B, num_classes=1 for regression
        super(Classifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.fc(x)
        return x

class SingleScanModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x 
    

class SingleScanModelBP(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModelBP, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Assuming x is a tensor of shape (batch_size, 2, C, D, H, W),
        # where 2 represents the two scans.
        # x.split(1, dim=1) will produce a tuple of tensors, 
        # each with shape (batch_size, 1, C, D, H, W).
        # The self.backbone expects input of shape (batch_size, C, D, H, W).
        
        scan_features_list = []
        for scan_tensor_with_extra_dim in x.split(1, dim=1):
            # Squeeze out the channel_dim (dim=1) which was of size 1
            squeezed_scan_tensor = scan_tensor_with_extra_dim.squeeze(1)
            feature = self.backbone(squeezed_scan_tensor)
            scan_features_list.append(feature)
        
        # scan_features_list now contains two tensors, e.g., [(B, 768), (B, 768)]
        
        # Stack these features along a new dimension (dim=1)
        # Resulting shape: (batch_size, 2, 768)
        stacked_features = torch.stack(scan_features_list, dim=1)
        
        # Perform mean pooling across the two scans (the new dim=1)
        # Resulting shape: (batch_size, 768)
        merged_features = torch.mean(stacked_features, dim=1)
        
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output 
    
class SingleScanModelQuad(nn.Module):
    """
    Model for quad image classification that processes four images through 
    shared backbone and merges their features.
    """
    def __init__(self, backbone, classifier):
        super(SingleScanModelQuad, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 4, C, D, H, W) - quad images
        Returns:
            output: Classification output
        """
        # Extract individual images
        image1 = x[:, 0]  # (batch_size, C, D, H, W)
        image2 = x[:, 1]  # (batch_size, C, D, H, W)
        image3 = x[:, 2]  # (batch_size, C, D, H, W)
        image4 = x[:, 3]  # (batch_size, C, D, H, W)
        
        # Process all images through shared backbone
        features1 = self.backbone(image1)  # (batch_size, embed_dim)
        features2 = self.backbone(image2)  # (batch_size, embed_dim)
        features3 = self.backbone(image3)  # (batch_size, embed_dim)
        features4 = self.backbone(image4)  # (batch_size, embed_dim)
        
        # Stack features and compute mean pooling
        # Resulting shape: (batch_size, 4, embed_dim) -> (batch_size, embed_dim)
        stacked_features = torch.stack([features1, features2, features3, features4], dim=1)
        merged_features = torch.mean(stacked_features, dim=1)
        
        # Apply dropout and classifier
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output 
    


# -----------------------
# Load backbone
# -----------------------
backbone = ViTBackboneNet("./checkpoints/BrainIAC.ckpt")
backbone.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)

# -----------------------
# Fonction preprocessing
# -----------------------
def preprocess_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()

    # Z-score
    data = (data - data.mean()) / (data.std() + 1e-8)

    # Convert to torch
    tensor = torch.tensor(data, dtype=torch.float32)

    # Add channel dimension 
    tensor = tensor.unsqueeze(0)

    # Resize (96,96,96)
    tensor = F.interpolate(
        tensor.unsqueeze(0),   
        size=(96,96,96),
        mode="trilinear",
        align_corners=False
    ).squeeze(0)  

    return tensor  # shape (1,96,96,96)

# -----------------------
# Extraction
# -----------------------
KDE_DIR = "./KDE_samples/KDE_samples"
nii_paths = sorted(glob(os.path.join(KDE_DIR, "pmid_*.nii.gz")))

print("=" * 100)
print("BRAINIAC EMBEDDING EXTRACTION")
print("=" * 100)
print(f"Processing directory: {KDE_DIR}")
print(f"Found {len(nii_paths)} NIfTI files")
print("=" * 100)

embeddings_with_pids = []

with torch.no_grad():
    for idx, path in enumerate(nii_paths, 1):
        # EXTRACT PID FROM FILENAME
        filename = os.path.basename(path)
        pid = filename.split('_')[1].split('.')[0]
        
        # Preprocess and extract embedding
        tensor = preprocess_nifti(path)
        tensor = tensor.unsqueeze(0).to(device)  
        embedding = backbone(tensor)  
        
        embeddings_with_pids.append((embedding.cpu(), pid))
        
        # Print progress
        print(f"[{idx}/{len(nii_paths)}] Processed: {filename} → PID: {pid}")

print("=" * 100)

# -----------------------
# SAVE INDIVIDUAL EMBEDDINGS (One file per PID)
# -----------------------
INDIVIDUAL_DIR = "results/individual_embeddings"
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

print("\n" + "=" * 100)
print("SAVING INDIVIDUAL EMBEDDINGS")
print("=" * 100)

for idx, (embedding, pid) in enumerate(embeddings_with_pids, 1):
    # Convert to numpy
    emb_np = embedding.numpy().flatten()
    
    # Save as .npy file (efficient for ML)
    npy_path = os.path.join(INDIVIDUAL_DIR, f"pid_{pid}_embedding.npy")
    np.save(npy_path, emb_np)
    
    # save as .csv for readability
    csv_path = os.path.join(INDIVIDUAL_DIR, f"pid_{pid}_embedding.csv")
    pd.DataFrame([emb_np], columns=[f'feature_{i}' for i in range(len(emb_np))]).to_csv(csv_path, index=False)
    
    print(f"[{idx}/{len(embeddings_with_pids)}] Saved: {npy_path} ({len(emb_np)} features)")

print("=" * 100)
print(f"All individual embeddings saved to: {INDIVIDUAL_DIR}/")
print(f"Files created: {len(embeddings_with_pids)} embedding files")
print("=" * 100)



# -----------------------
# SUMMARY
# -----------------------
print("\n" + "=" * 100)
print("EXTRACTION COMPLETE")
print("=" * 100)
print(f"Individual embeddings: {INDIVIDUAL_DIR}/")
print(f"   - {len(embeddings_with_pids)} .npy files (for ML loading)")
print(f"   - {len(embeddings_with_pids)} .csv files (for human readability)")
print(f"Total samples: {len(embeddings_with_pids)}")
print(f"Embedding dimensions: {embeddings_with_pids[0][0].shape[1]}")
print("=" * 100)





