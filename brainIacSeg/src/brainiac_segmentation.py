import os
import yaml
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, ScaleIntensityRanged, ToTensord
)
from segmentation_model import ViTUNETRSegmentationModel

# 1. Détecter le matériel une bonne fois pour toutes
device = torch.device("cpu") 

def load_brainiac_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialisation de l'architecture
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=checkpoint_path, 
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    print("="*30)
    print("Succès : Poids Backbone (ViT) chargés depuis BrainIAC.ckpt")
    print("Note : Le décodeur est aléatoire. Le résultat sera du bruit.")
    print("="*30)
    
    # Correction : On envoie au device (CPU) et on ne fait pas .cuda()
    return model.eval().to(device), config

def get_transforms(config):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ])

def segment_image(image_path, output_path, model, config):
    transforms = get_transforms(config)
    data = {"image": image_path}
    data_transformed = transforms(data)
    
    # Correction : .to(device) au lieu de .cuda()
    input_tensor = data_transformed["image"].unsqueeze(0).to(device)

    print(f"Inférence sur {device} en cours (patience...)...")
    
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=tuple(config['model']['img_size']),
            sw_batch_size=1, # Plus sûr pour la RAM de ton PC
            predictor=model,
            overlap=0.5
        )
        output = torch.sigmoid(output)
        mask = (output > 0.5).float()
        
    mask_np = mask.squeeze().cpu().numpy()
    original_nii = nib.load(image_path)
    
    # On crée le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_nii = nib.Nifti1Image(mask_np.astype(np.uint8), original_nii.affine)
    nib.save(result_nii, output_path)
    print(f"Terminé ! Résultat : {output_path}")

if __name__ == "__main__":
    PATH_CONFIG = "./config/config.yml"
    PATH_CHECKPOINT = "../data/weights/BrainIAC.ckpt"
    IMAGE_A_SEGMENTER = "../data/KDE_samples/pmid_21273134.nii.gz" # Attention au .gzz corrigé en .gz
    NOM_SORTIE = "./segmentation_results/output_seg.nii.gz"

    try:
        model, config = load_brainiac_model(PATH_CONFIG, PATH_CHECKPOINT)
        segment_image(IMAGE_A_SEGMENTER, NOM_SORTIE, model, config)
    except Exception as e:
        print(f"Erreur : {e}")