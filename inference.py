import argparse
import librosa

from coco_mulla.models import CoCoMulla ## Change this one!!
from coco_mulla.utilities import *
from coco_mulla.utilities.encodec_utils import save_rvq

from coco_mulla.utilities.sep_utils import separate
from config import TrainCfg  ## change this one!!
import torch.nn.functional as F
import numpy as np 
import torch
from scipy.interpolate import CubicSpline

device = get_device()

def interpolate(data, pad = False):
    dim = data.shape[1]
    data_new = np.zeros((500, dim))
    seq_len = data.shape[0]
    # Original time points (video sequence length)
    t_original = np.linspace(0, 1, seq_len)
    # New time points (target sequence length)
    t_new = np.linspace(0, 1, 500)
    for i in range(dim):  # Iterate over features
        cs = CubicSpline(t_original, data[:, i])
        data_new[:, i] = cs(t_new)
    if pad:
        pad_frame = np.zeros((1, data_new.shape[1]), dtype=data_new.dtype)
        data_new = np.concatenate([data_new, pad_frame], axis=0)

    return data_new


def generate(model_path, batch, is_video, is_motion, is_face):
    model = CoCoMulla(
        sec=TrainCfg.sample_sec,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        is_video=is_video,
        is_motion=is_motion,
        is_face=is_face
    ).to(device)
    model.load_weights(model_path)
    model.eval()
    print(batch)
    with torch.no_grad():
        gen_tokens = model(**batch)

    return gen_tokens



def load_data(video_path=None, face_path=None, motion_path=None, offset=0):
    video_emb, face_emb, motion_emb = None, None, None

    # Load video if path is provided
    if video_path is not None:
        video_tmp = torch.load(video_path).to(torch.float32).detach().cpu().numpy()
        video_tmp = interpolate(video_tmp, pad=True)
        video_emb = torch.from_numpy(video_tmp).to(device).float()

    # Load face if path is provided
    if face_path is not None:
        face_tmp = np.load(face_path)
        face_tmp = interpolate(face_tmp, pad=True)
        face_emb = torch.from_numpy(face_tmp).to(device).float()

    # Load motion if path is provided
    if motion_path is not None:
        motion_tmp = np.load(motion_path)
        motion_tmp = motion_tmp[:, 0, :]
        motion_tmp = interpolate(motion_tmp, pad=True)
        motion_emb = torch.from_numpy(motion_tmp).to(device).float()

    print("Loaded shapes:",
          f"video={video_emb.shape if video_emb is not None else None},",
          f"face={face_emb.shape if face_emb is not None else None},",
          f"motion={motion_emb.shape if motion_emb is not None else None}")
    return video_emb, face_emb, motion_emb

def save_pred(output_folder, tags, pred):
    mkdir(output_folder)
    output_list = [os.path.join(output_folder, tag) for tag in tags]
    save_rvq(output_list=output_list, tokens=pred)


def wrap_batch(video=None, face=None, motion=None, prompt=""):
    num_samples = 1
    batch = {
        "music": None,
        "desc": [prompt] * num_samples,
        "num_samples": num_samples,
        "mode": "inference",
        "video":None,
        "face":None,
        "motion":None
    }

    # Repeat & attach only if not None
    if video is not None:
        batch["video"] = video.repeat(num_samples, 1, 1)
    if face is not None:
        batch["face"] = face.repeat(num_samples, 1, 1)
    if motion is not None:
        batch["motion"] = motion.repeat(num_samples, 1, 1)

    return batch


def inference(args):
    video,face, motion = load_data(video_path=args.video_path,
                                   face_path=args.face_path,
                                   motion_path=args.motion_path,
                                       offset=args.offset)
    # Decide which flags are True/False
    is_video = (video is not None)
    is_motion = (motion is not None)
    is_face = (face is not None)
    print('is video, is_motion, is_face', is_video, is_motion, is_face)

    batch = wrap_batch(video, face, motion, read_lst(args.prompt_path)[0])
    pred = generate(model_path=args.model_path,
                    batch=batch,
                    is_video=is_video,
                    is_motion=is_motion,
                    is_face=is_face)
    save_pred(output_folder=args.output_folder,
              tags=['generated'],
              pred=pred)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference script for CoCoMulla.")
    parser.add_argument("--num_layers", type=int, default=48, help="Number of transformer layers.")
    parser.add_argument("--latent_dim", type=int, default=48, help="Latent dimension size.")
    parser.add_argument("--output_folder", type=str, help="Where to store the generated outputs.")
    parser.add_argument("--model_path", type=str,help="Path to the model checkpoint.")
    parser.add_argument("--prompt_path", type=str,help="Path to a text file containing your generation prompt.")
    parser.add_argument("--video_path", type=str, help="Path to video embedding, if available.")
    parser.add_argument("--motion_path", type=str, help="Path to motion embedding, if available.")
    parser.add_argument("--face_path", type=str, help="Path to face embedding, if available.")
    parser.add_argument("--offset", type=int, default=0,help="Offset in frames or seconds (if needed).")

    args = parser.parse_args()
    inference(args)