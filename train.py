#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from coco_mulla.data_loader.dataset_sampler import Dataset, collate_fn
from config import TrainCfg
from coco_mulla.models import CoCoMulla
from coco_mulla.utilities.trainer_utils import Trainer

device = "cuda"  # or "cpu" if you prefer

###############################################################################
# ARGUMENT PARSING
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train script for CoCoMulla.")
    parser.add_argument("-l", "--lst-path", type=str, required=True, help="Path to your .lst file")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to store logs/checkpoints.")

    # NEW: Booleans to enable or disable each modality
    parser.add_argument("--is-video", action="store_true", help="If set, load video data")
    parser.add_argument("--is-motion", action="store_true", help="If set, load motion data")
    parser.add_argument("--is-face", action="store_true", help="If set, load face data")

    return parser.parse_args()

###############################################################################
# LOSS FUNCTION
###############################################################################
def loss_fn(outputs, y):
    """
    Standard cross-entropy on the predicted logits, 
    masked by 'outputs.mask'.
    """
    prob = outputs.logits
    mask = outputs.mask

    # Only keep positions where mask == True
    prob = prob[mask]   # [N_masked, vocab_size]
    y = y[mask]         # [N_masked]

    # The 2048 can be adjusted if your model uses a different codebook size
    prob = prob.view(-1, 2048)
    return nn.CrossEntropyLoss()(prob, y)

###############################################################################
# TRAINING LOOP
###############################################################################
def train(model, dataset, dataloader, device, model_dir, learning_rate, is_video, is_motion, is_face):
    out_name = '1' if is_video else ''
    out_name += '2' if is_motion else ''
    out_name += '3' if is_face else ''
    print(out_name)
    # Hyperparams & setup
    num_steps = len(dataloader)
    epochs = TrainCfg.epoch
    rng = np.random.RandomState(569)  # you can seed as you like

    # Create summary writer for TensorBoard
    writer = SummaryWriter(model_dir, flush_secs=20)

    # Initialize optimizer & LR scheduler
    trainer = Trainer(
        params=model.parameters(),
        lr=learning_rate,
        num_epochs=epochs,
        num_steps=num_steps
    )

    model = model.to(device)

    step = 0
    for e in range(1, epochs+1):
        mean_loss = 0.0
        n_element = 0

        model.train()
        dl = tqdm(dataloader, desc=f"Epoch {e}")

        # Reset dataset seed each epoch
        r = rng.randint(0, 233333)
        dataset.reset_random_seed(r, e)

        for i, batch in enumerate(dl):
            desc = batch["desc"]                           # Always present
            music = batch["music"].to(device).long()       # Always present

            # Check optional keys
            if is_video:
                video = batch["video"].to(device).float()
            else:
                video = None

            if is_motion:
                motion = batch["motion"].to(device).float()
            else:
                motion = None

            if is_face:
                face = batch["face"].to(device).float()
            else:
                face = None

            # Build input dict for model
            batch_1 = {
                "music":  music,
                "desc":   desc,
                "video":  video,
                "motion": motion,
                "face":   face
            }

            # Forward pass (autocast optional)
            outputs = model(**batch_1)

            # Compute loss
            r_loss = loss_fn(outputs, music)

            # Backprop & optimizer step
            grad_1, lr_1 = trainer.step(r_loss, model.parameters())

            step += 1
            n_element += 1
            mean_loss += r_loss.item()

            # Logging
            writer.add_scalar("r_loss", r_loss.item(), step)
            writer.add_scalar("grad_1", grad_1, step)
            writer.add_scalar("lr_1", lr_1, step)

        # Epoch stats
        mean_loss = mean_loss / n_element if n_element > 0 else 0.0
        writer.add_scalar('train/mean_loss', mean_loss, step)
        # if e % 10 == 0:
        # Save checkpoint
        with torch.no_grad():
            model.save_weights(os.path.join(model_dir, f"diff_{e}_{out_name}.pth"))

###############################################################################
# MAIN
###############################################################################
def main():
    args = parse_args()

    model_dir = args.output_dir+'/models/'

    # Ensure output directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Prepare dataset & dataloader
    dataset = Dataset(
        rid=0,
        path_lst=[args.lst_path],
        sampling_prob=None,
        sampling_strategy='prob-based',
        cfg=TrainCfg,
        # NEW: pass booleans to dataset
        is_video=args.is_video,
        is_motion=args.is_motion,
        is_face=args.is_face
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TrainCfg.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # Prepare model
    learning_rate = 0.05
    num_layers = 48
    latent_dim = 48
    model = CoCoMulla(TrainCfg.sample_sec, num_layers=num_layers, latent_dim=latent_dim, 
                       is_video=args.is_video,
                        is_motion=args.is_motion,
                        is_face=args.is_face
                      ).to(device)
    model.set_training()

    # Run training
    train(model, dataset, dataloader, device, model_dir, learning_rate, is_video=args.is_video,
        is_motion=args.is_motion,
        is_face=args.is_face)

if __name__ == "__main__":
    main()
