import os
from coco_mulla.utilities.encodec_utils import extract_rvq
import librosa
import torch
import numpy as np
import tqdm
import argparse 

def compute_rvq(folder_path,output_path):
    sr=32000
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(folder_path)
    for i in os.listdir(folder_path):
        i_ = os.path.join(folder_path, i)
        wav, _ = librosa.load(i_, sr=sr, mono=True)
        wav = torch.from_numpy(wav).to(device)[None, None, ...]
        mix_rvq = extract_rvq(wav, sr=sr)
        print(mix_rvq.shape)
        np.save(os.path.join(output_path,i.replace(".mp3",".npy")), mix_rvq.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--folder_path', type=str)
    parser.add_argument('-o', '--output_folder', type=str)
    args = parser.parse_args()
    compute_rvq(args.folder_path, args.output_folder)