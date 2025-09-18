from __future__ import annotations

import os
import torch
import numpy as np
from transformers import AutoTokenizer
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModel3D
from medclip.dataset import MRIBiomarkerDataset


def load_trained_model(checkpoint_path: str):
    """Load the trained 3D MedCLIP model from checkpoint."""
    model = MedCLIPModel(vision_cls=MedCLIPVisionModel3D)
    
    # Load the trained weights
    state_dict = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'))
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    
    return model


def prepare_mri_volume(mri_path: str, volume_size=(96, 128, 128)):
    """Load and preprocess a single MRI volume."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError('nibabel is required to load NIfTI volumes')
    
    # Load volume
    vol = nib.load(mri_path).get_fdata().astype(np.float32)
    
    # Normalize per volume
    if np.nanstd(vol) > 0:
        vol = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-6)
    vol = np.nan_to_num(vol)
    
    # Resize if requested
    if volume_size is not None:
        t = torch.from_numpy(vol)[None, None]
        t = torch.nn.functional.interpolate(t, size=volume_size, mode='trilinear', align_corners=False)
        vol = t[0, 0].numpy()
    
    # To tensor [N=1, C=1, D, H, W]
    vol_t = torch.from_numpy(vol)[None, None].cuda()
    
    # Repeat to 3 channels for 3D ResNet
    if vol_t.shape[1] == 1:
        vol_t = vol_t.repeat((1, 3, 1, 1, 1))
    
    return vol_t


def prepare_text(text: str):
    """Tokenize text for the model."""
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.model_max_length = 77
    
    inputs = tokenizer([text], truncation=True, padding=True, return_tensors='pt')
    return inputs['input_ids'].cuda(), inputs['attention_mask'].cuda()


def compute_similarity(model, mri_volume, text_input_ids, text_attention_mask):
    """Compute image-text similarity."""
    with torch.no_grad():
        # Encode image and text
        img_embeds = model.encode_image(mri_volume)
        text_embeds = model.encode_text(text_input_ids, text_attention_mask)
        
        # Compute similarity
        similarity = torch.matmul(img_embeds, text_embeds.T)
        
        return similarity.item()


def main():
    # Load trained model
    checkpoint_path = './checkpoints/ad-clip3d'
    model = load_trained_model(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    
    # Example 1: Single MRI with biomarker text
    mri_path = './Data/2017-07-13_13_02_56I874250_3_accelerated_sag_ir-fspgr.nii.gz'
    biomarker_text = "DX: MCI. Plasma biomarkers: pTau181=8.25."
    
    if os.path.exists(mri_path):
        # Prepare inputs
        mri_volume = prepare_mri_volume(mri_path)
        text_input_ids, text_attention_mask = prepare_text(biomarker_text)
        
        # Compute similarity
        similarity = compute_similarity(model, mri_volume, text_input_ids, text_attention_mask)
        
        print(f"\nExample 1:")
        print(f"MRI: {os.path.basename(mri_path)}")
        print(f"Text: {biomarker_text}")
        print(f"Similarity: {similarity:.4f}")
    
    # Example 2: Compare multiple biomarker descriptions
    biomarker_texts = [
        "DX: Normal. Plasma biomarkers: pTau181=2.0.",
        "DX: MCI. Plasma biomarkers: pTau181=8.25.",
        "DX: AD. Plasma biomarkers: pTau181=15.0.",
    ]
    
    if os.path.exists(mri_path):
        mri_volume = prepare_mri_volume(mri_path)
        
        print(f"\nExample 2: Comparing biomarker descriptions for {os.path.basename(mri_path)}")
        similarities = []
        
        for text in biomarker_texts:
            text_input_ids, text_attention_mask = prepare_text(text)
            similarity = compute_similarity(model, mri_volume, text_input_ids, text_attention_mask)
            similarities.append(similarity)
            print(f"Text: {text}")
            print(f"Similarity: {similarity:.4f}")
        
        # Find best match
        best_idx = np.argmax(similarities)
        print(f"\nBest match: {biomarker_texts[best_idx]} (similarity: {similarities[best_idx]:.4f})")
    
    # Example 3: Batch inference on multiple MRIs
    print(f"\nExample 3: Batch inference")
    data_dir = './Data'
    csv_path = './adni_apoe_plasma_one_row_has_ptau.csv'
    
    if os.path.exists(csv_path) and os.path.exists(data_dir):
        # Create dataset for batch processing
        dataset = MRIBiomarkerDataset(
            csv_path=csv_path,
            mri_dir=data_dir,
            volume_size=(96, 128, 128),
        )
        
        # Process first 5 samples
        batch_size = 5
        for i in range(min(batch_size, len(dataset))):
            sample = dataset[i]
            mri_vol = sample['pixel_values'].cuda()
            # Ensure 5D tensor [N, C, D, H, W]
            if mri_vol.dim() == 4:
                mri_vol = mri_vol.unsqueeze(0)  # Add batch dimension
            if mri_vol.shape[1] == 1:
                mri_vol = mri_vol.repeat((1, 3, 1, 1, 1))
            
            text_ids = sample['input_ids'].cuda()
            text_mask = sample['attention_mask'].cuda()
            
            similarity = compute_similarity(model, mri_vol, text_ids.unsqueeze(0), text_mask.unsqueeze(0))
            
            print(f"Sample {i+1}: Similarity = {similarity:.4f}")


if __name__ == '__main__':
    main()
