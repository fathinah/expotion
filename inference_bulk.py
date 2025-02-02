#!/usr/bin/env python3

import os
import argparse
import torch
import numpy as np
from scipy.interpolate import CubicSpline
from coco_mulla.utilities.encodec_utils import save_rvq

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS (from your original code)
# -----------------------------------------------------------------------------

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

device = get_device()

def interpolate(data, pad=False):
    dim = data.shape[1]
    data_new = np.zeros((500, dim))
    seq_len = data.shape[0]
    # Original time points
    t_original = np.linspace(0, 1, seq_len)
    t_new = np.linspace(0, 1, 500)
    for i in range(dim):
        cs = CubicSpline(t_original, data[:, i])
        data_new[:, i] = cs(t_new)
    if pad:
        pad_frame = np.zeros((1, data_new.shape[1]), dtype=data_new.dtype)
        data_new = np.concatenate([data_new, pad_frame], axis=0)
    return data_new


def replicate_frames(data, factor=10):
    """
    Replicate each frame 'factor' times along the time axis.
    E.g. a (50, dim) array -> (500, dim) array if factor=10.
    """
    data_new = np.repeat(data, repeats=factor, axis=0)
    pad_frame = np.zeros((1, data_new.shape[1]), dtype=data_new.dtype)
    data_new = np.concatenate([data_new, pad_frame], axis=0)
    return data_new

def read_lst(lst_path):
    with open(lst_path, "r") as f:
        lines = [x.strip() for x in f.readlines()]
    return lines

def mkdir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# LOAD DATA FUNCTION
# -----------------------------------------------------------------------------

def load_data(video_path=None, face_path=None, motion_path=None, offset=0):
    """
    Loads and interpolates the video/motion/face data if the paths exist.
    Returns (video_emb, face_emb, motion_emb), any of which can be None.
    """
    video_emb = None
    face_emb = None
    motion_emb = None

    if video_path and os.path.isfile(video_path):
        video = torch.load(video_path).float().cpu().numpy()
        # video = replicate_frames(video)
        video_emb = torch.from_numpy(video).to(device).float()

    if face_path and os.path.isfile(face_path):
        face = torch.load(face_path)
        # face = replicate_frames(face)
        face_emb = face['face_p'].to(device).float()

    if motion_path and os.path.isfile(motion_path):
        motion = np.load(motion_path)
        # e.g. shape [T, 1, 34], so we take motion[:, 0, :]
        if motion.ndim == 3:
            motion = motion[:,0,:]
        # motion = replicate_frames(motion)
        motion_emb = torch.from_numpy(motion).to(device).float()

    print(f" Loaded shapes: video={video_emb.shape if video_emb is not None else None},"
          f" face={face_emb.shape if face_emb is not None else None},"
          f" motion={motion_emb.shape if motion_emb is not None else None}")
    return video_emb, face_emb, motion_emb

# -----------------------------------------------------------------------------
# WRAP BATCH
# -----------------------------------------------------------------------------
def wrap_batch(video=None, face=None, motion=None, prompt=""):
    """
    Build the batch dict dynamically with whichever tensors are not None.
    """
    batch = {
        "music": None,
        "desc": [prompt],  # single sample
        "num_samples": 1,
        "mode": "inference",
        "video":None,
        "face":None,
        "motion":None
    }
    if video is not None:
        batch["video"] = video.unsqueeze(0)  # shape [1, T, D]
    if face is not None:
        batch["face"] = face.unsqueeze(0)
    if motion is not None:
        batch["motion"] = motion.unsqueeze(0)
    return batch

# -----------------------------------------------------------------------------
# MODEL INFERENCE WRAPPER
# -----------------------------------------------------------------------------
def generate(model, batch):
    """
    This calls the forward pass of your CoCoMulla model in 'inference' mode
    and returns the tokens.
    """
    model.eval()
    with torch.no_grad():
        tokens = model(**batch)
    return tokens

def save_pred(output_folder, tags, pred):
    mkdir(output_folder)
    output_list = [os.path.join(output_folder, tag) for tag in tags]
    save_rvq(output_list=output_list, tokens=pred)

# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Bulk inference for multiple chunk_X in a folder.")
    parser.add_argument("--num_layers", type=int, default=48, help="Number of layers.")
    parser.add_argument("--latent_dim", type=int, default=48, help="Latent dimension.")
    parser.add_argument("--output_folder", type=str, required=True, help="Where to store outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--prompt_path", type=str, required=False, help="Path to file with prompt text.")
    parser.add_argument("--chunks_folder", type=str, required=True,
                        help="Folder containing subdirectories named chunk_XXX.")
    parser.add_argument("--offset", type=int, default=0, help="Offset for data loading if needed.")

     # NEW: Booleans to enable or disable each modality
    parser.add_argument("--is-video", action="store_true", help="If set, load video data")
    parser.add_argument("--is-motion", action="store_true", help="If set, load motion data")
    parser.add_argument("--is-face", action="store_true", help="If set, load face data")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# MAIN INFERENCE LOGIC
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # 1) Prepare the model (loaded ONCE)
    from coco_mulla.models import CoCoMulla
    from config import TrainCfg

    # Adjust these flags if needed: is_video, is_motion, is_face
    # If you truly want them dynamic, you can guess them from the presence of files,
    # or pass them as additional flags. For simplicity, let's turn them all on:
    is_video = args.is_video
    is_motion = args.is_motion 
    is_face = args.is_face

    print(f"[INFO] Loading model from: {args.model_path}")
    model = CoCoMulla(
        sec=TrainCfg.sample_sec,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        is_video=is_video,
        is_motion=is_motion,
        is_face=is_face
    ).to(device)
    model.load_weights(args.model_path)
    print("[INFO] Model loaded successfully.")

    # 2) Read prompt text
    prompt_str = "Default prompt"
    if args.prompt_path and os.path.isfile(args.prompt_path):
        lines = read_lst(args.prompt_path)
        if lines:
            prompt_str = lines[0]
    print(f"[INFO] Using prompt: {prompt_str}")

    # 3) Iterate over each chunk_X subdir in args.chunks_folder
    folder = args.chunks_folder
    if not os.path.isdir(folder):
        raise ValueError(f"chunks_folder '{folder}' is not a valid directory.")

    # Make sure output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    # Loop through subdirs
    for subdir in sorted(os.listdir(folder)):
        print(subdir)
        # Only process subdirectories named "chunk_..."
        # if not subdir.startswith("chunk_"):
        #     continue

        chunk_dir = os.path.join(folder, subdir)
        # if not os.path.isdir(chunk_dir):
        #     continue

        # print(f"========= Processing {chunk_dir} =========")
        # 4) Build paths to video, face, motion
        video_path, motion_path, face_path = None, None, None
        if is_video:
            video_path = os.path.join(chunk_dir, "resnet.npy")
        if is_motion:
            motion_path = os.path.join(chunk_dir, "motion_kpts.npy")
            if not os.path.exists(motion_path):
                continue
        if is_face:
            face_path   = chunk_dir ##os.path.join(chunk_dir, "face.npy")
            print(face_path)

        # 5) Load data
        video_emb, face_emb, motion_emb = load_data(
            video_path=video_path,
            face_path=face_path,
            motion_path=motion_path,
            offset=args.offset
        )

        

        # 6) Wrap batch
        batch = wrap_batch(video=video_emb, face=face_emb, motion=motion_emb, prompt=prompt_str)

        print('batch')
        print(batch)

        # 7) Generate
        pred = generate(model=model, batch=batch)

        # 8) Save
        tag_name = f"{subdir}_generated"
        save_pred(args.output_folder, [tag_name], pred)

        print(f"Saved tokens for {subdir} -> {args.output_folder}/{tag_name}.wav")

    print("All chunks processed. Done!")

if __name__ == "__main__":
    main()
