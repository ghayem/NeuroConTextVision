import os
import torch
import nibabel as nib
import numpy as np
import argparse
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, NormalizeIntensityd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from segmentation_model import ViTUNETRSegmentationModel

# Device-agnostic setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_inference(config, state_dict):
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=config['pretrain']['simclr_checkpoint_path'],
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[len('model.'):]
        
        # KEY REMAPPING: Redirect vit keys to unetr.vit
        if k.startswith('vit.'):
            k = 'unetr.' + k
            
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    return model.eval().to(device)

def preprocess_image(image_path, config):
    img_size = tuple(config['model']['img_size'])
    transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        Resized(keys=['image'], spatial_size=img_size, mode='trilinear'),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        EnsureTyped(keys=['image'])
    ])
    
    data = transforms({'image': image_path})
    meta = data['image'].meta if hasattr(data['image'], 'meta') else data['image_meta_dict']
    return data['image'].unsqueeze(0).to(device), meta

def generate_segmentation(model, image_tensor, config):
    """
    Runs inference and returns the segmentation mask.
    """
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=image_tensor,
            roi_size=tuple(config['model']['img_size']),
            sw_batch_size=config['training']['sw_batch_size'],
            predictor=model,
            overlap=0.5
        )
    
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    # FIX: Manually remove the batch dimension instead of using decollate_batch
    # pred shape is (Batch, Channels, H, W, D) -> e.g., (1, 1, 96, 96, 96)
    # We want (Channels, H, W, D)
    if pred.shape[0] == 1:
        pred = pred[0]
    else:
        # Fallback for unexpected batch sizes
        from monai.data import decollate_batch
        pred = decollate_batch(pred)[0]
        
    return pred

def save_segmentation(segmentation_tensor, meta_dict, output_path):
    original_path = meta_dict.get('filename_or_obj', '')
    if original_path and os.path.exists(original_path):
        ref_nii = nib.load(original_path)
        affine = ref_nii.affine
        header = ref_nii.header
    else:
        affine = np.eye(4)
        header = None

    seg_np = segmentation_tensor.cpu().numpy()
    if seg_np.shape[0] == 1:
        seg_np = seg_np[0]

    nii_img = nib.Nifti1Image(seg_np.astype(np.float32), affine=affine, header=header)
    nib.save(nii_img, output_path)
    print(f"Segmentation saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--simclr_checkpoint_path', type=str, default=None)
    parser.add_argument('--gpu_device', type=str, default="0")
    args = parser.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
        print(f"Using GPU: {args.gpu_device}")
    else:
        print("CUDA not available. Running on CPU.")

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']

    if args.simclr_checkpoint_path:
        config['pretrain']['simclr_checkpoint_path'] = args.simclr_checkpoint_path

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model_for_inference(config, state_dict)
    image_tensor, meta_dict = preprocess_image(args.image_path, config)
    seg_tensor = generate_segmentation(model, image_tensor, config)
    
    name = os.path.basename(args.image_path).split('.')[0]
    output_path = os.path.join(args.output_dir, f"{name}_seg.nii.gz")
    save_segmentation(seg_tensor, meta_dict, output_path)
    print("Done.")