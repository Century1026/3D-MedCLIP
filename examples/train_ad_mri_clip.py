from __future__ import annotations

import os
from torch.utils.data import DataLoader
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModel3D
from medclip.dataset import MRIBiomarkerDataset, MRIBiomarkerCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer


def main():
    data_dir = './Data'
    csv_path = './adni_apoe_plasma_one_row_has_ptau.csv'

    # Dataset and loader
    dataset = MRIBiomarkerDataset(
        csv_path=csv_path,
        mri_dir=data_dir,
        volume_size=(96, 128, 128),
    )
    collator = MRIBiomarkerCollator()
    trainloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
    )

    # Model
    model = MedCLIPModel(vision_cls=MedCLIPVisionModel3D)
    model.cuda()

    # Loss and training
    loss_model = ImageTextContrastiveLoss(model)
    loss_model.cuda()

    trainer = Trainer()
    trainer.train(
        model,
        train_objectives=[(trainloader, loss_model, 1)],
        epochs=10,
        warmup_ratio=0.01,
        optimizer_params={'lr': 2e-5},
        output_path='./checkpoints/ad-clip3d',
        evaluation_steps=100,
        save_steps=1000,
        use_amp=True,
    )


if __name__ == '__main__':
    main()

