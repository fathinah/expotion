#!/usr/bin/env python3
"""
Batch Inference Script for Face-Only Inference

This script processes one or more videos in batch. The --video_path argument
can be a single video file path or a .lst file containing multiple video paths,
one per line.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python inference_face_only_batch.py \
      --num_layers=48 --latent_dim=12 \
      --output_folder="demo/output/video_0318" \
      --video_path="/path/to/video_list.lst" \
      --model_path="/path/to/model/diff_49_end.pth" \
      --prompt_path="/path/to/captions.json" \
      --offset=0
"""
from tqdm import tqdm
import argparse
import os
import shlex
import subprocess
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import dlib
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
import soundfile as sf

# Import your models and utilities
from marlin_pytorch import Marlin
from coco_mulla.models import CoCoMulla
from coco_mulla.utilities import *
from coco_mulla.utilities.encodec_utils import extract_rvq, save_rvq
from coco_mulla.utilities.symbolic_utils import process_midi, process_chord
from coco_mulla.utilities.sep_utils import separate
from config import TrainCfg

# If not already defined, a helper to read .lst files
def read_lst(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Helper to interpret the --video_path argument:
# If it ends with .lst, treat it as a file with multiple paths; otherwise, a single video path.
def read_video_paths(video_path_arg):
    if video_path_arg.endswith('.lst'):
        return read_lst(video_path_arg)
    else:
        return [video_path_arg]

# Device handling
DEVICE = get_device()
device = get_device()

def mark_face_video(input_video_path):
    print("Marking landmarks")
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    gpu_ids = [1, 2]
    cap = cv2.VideoCapture(input_video_path)
    output_video_path = f'{input_video_path[:-4]}_marked.mp4'
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / 5)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count % frame_skip == 0:
            if frame_count % 2 == 0:
                torch.cuda.set_device(gpu_ids[0])
            else:
                torch.cuda.set_device(gpu_ids[1])
            faces = cnn_face_detector(gray, 1)
            for face in faces:
                rect = face.rect
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Saved to {output_video_path}")
    return output_video_path

def resample(input_path, output_path, rate=5*16):
    command = f"ffmpeg -i {input_path} -vf fps={rate} -y {output_path}"
    command_list = shlex.split(command)
    subprocess.run(command_list)

def compute_feature_face(model, video):
    features = model.extract_video(video, sample_rate=1, stride=16,
                                   crop_face=True, detector_device=DEVICE)
    print("Feature shape:", features.shape)
    return features

def generate(model, batch, args):
    with torch.no_grad():
        gen_tokens = model(**batch)
        gen_audio = model.musicgen.compression_model.decode(gen_tokens, None)
    return gen_tokens, gen_audio

def generate_mask(xlen):
    names = ["face", "face-body", "face-vid", "face-body-vid"]
    mask = torch.zeros([4, 2, xlen]).to(device)
    mask[1, 1] = 1
    mask[2, 0] = 1
    mask[3] += 1
    return mask, names

def load_data(video_path, offset):
    sr = TrainCfg.sample_rate
    res = TrainCfg.frame_res
    sample_sec = TrainCfg.sample_sec
    vp_resampled = f"{video_path[:-4]}_resampled.mp4"
    resample(video_path, vp_resampled)
    face_model = Marlin.from_online("marlin_vit_base_ytf").to(DEVICE)
    face = compute_feature_face(face_model, vp_resampled)
    T, _ = face.shape
    face_p = face.permute(1, 0).unsqueeze(0)
    new_T = T * 10
    face_p = F.interpolate(face_p, size=new_T, mode='linear', align_corners=True)
    face_p = face_p.squeeze(0).permute(1, 0)
    face_p = face_p.cpu().numpy()

    vid = np.zeros((new_T, 1))
    body = np.zeros((new_T, 1))

    face = crop(face_p[None, ...], sample_sec, res, offset=offset)
    body = crop(body[None, ...], sample_sec, res, offset=offset)
    vid = crop(vid[None, ...], sample_sec, res, offset=offset)
    print("face shape:", face_p.shape)
    print("vid shape:", vid.shape)
    print("body shape:", body.shape)
    face = torch.from_numpy(face).to(device)
    body = torch.from_numpy(body).to(device).float()
    vid = torch.from_numpy(vid).to(device).float()
    return face, body, vid

def crop(x, sample_sec, res, offset=0):
    xlen = x.shape[1]
    sample_len = int(sample_sec * res) + 1
    if xlen < sample_len:
        x = np.pad(x, ((0, 0), (0, sample_len - xlen), (0, 0)))
        return x
    st = offset * res
    ed = int((offset + sample_sec) * res) + 1
    print("Crop indices:", st, ed, "out of", x.shape[1])
    assert x.shape[1] > st
    return x[:, st:ed]

def replace_audio_in_video(video_path, audio_path, output_path):
    video_clip = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video_with_new_audio = video_clip.set_audio(new_audio)
    video_with_new_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

def save_pred(video_path, model_name, output_folder, wav, tags, pred):
    video_id = os.path.basename(video_path)
    print("videoid", video_id)
    ten_seconds_samples = 32000 * 10
    output_folder = os.path.join(output_folder, model_name)
    mkdir(output_folder)
    for i in range(wav.shape[0]):
        audio_p = os.path.join(output_folder, f'{video_id[-4]}_{tags[i]}.wav')
        wav_ = wav[i].squeeze().cpu()
        if len(wav_) > ten_seconds_samples:
            wav_ = wav_[:ten_seconds_samples]
        print("Audio shape:", wav_.shape)
        sf.write(audio_p, wav_, 32000)
        # out_video = os.path.join(output_folder, os.path.basename(video_path)[:-4] + f"_ours_{tags[i]}.mp4")
        # replace_audio_in_video(video_path, audio_p, out_video)

def wrap_batch(face, body, vid, cond_mask, prompt):
    num_samples = len(cond_mask)
    print('Number of samples:', num_samples)
    face = face.repeat(num_samples, 1, 1)
    body = body.repeat(num_samples, 1, 1)
    vid = vid.repeat(num_samples, 1, 1)
    prompt = [prompt] * num_samples
    batch = {
        "seq": None,
        "num_samples": num_samples,
        "desc": prompt,
        "body": body,
        "face": face,
        "vid": vid,
        "cond_mask": cond_mask,
        "mode": "inference",
    }
    return batch

def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def inference(args, model):
    face, body, vid = load_data(video_path=args.video_path, offset=args.offset)
    cond_mask, names = generate_mask(face.shape[-2])
    idd = os.path.basename(args.video_path).split('.')[0]
    if args.prompt_path.split('.')[-1] == 'json':
        captions = read_json(args.prompt_path)
        prompt = captions[idd].replace('<s>', '').replace('</s>', '')
    elif args.prompt_path.split('.')[-1] == 'txt':
        prompt = read_lst(args.prompt_path)[0]
    else:
        prompt = ""
    batch = wrap_batch(face, body, vid, cond_mask, prompt)
    # print("Batch shapes:", batch['face'].shape, batch['body'].shape, batch['vid'].shape, batch['cond_mask'].shape)
    # print("Description:", batch['desc'])
    pred, wav = generate(model=model, batch=batch, args=args)
    save_pred(video_path=args.video_path, model_name = args.model_path.split('/')[-2], output_folder=args.output_folder, wav=wav, tags=names, pred=pred)

def main(args):
    print("Sample seconds:", TrainCfg.sample_sec)
    model = CoCoMulla(TrainCfg.sample_sec,
                      num_layers=args.num_layers,
                      latent_dim=args.latent_dim).to(device)
    model.load_weights(args.model_path)
    model.eval()
    video_paths = read_video_paths(args.video_path)
    for video in tqdm(video_paths):
        print("Processing video:", video)
        # Update args.video_path for each iteration
        args.video_path = os.path.join(args.dataset_folder, video)
        inference(args, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-n', '--num_layers', type=int, required=True)
    parser.add_argument('-l', '--latent_dim', type=int, required=True)
    parser.add_argument('-a', '--video_path', type=str, required=True,
                        help="Single video path or a .lst file containing multiple relative video paths")
    parser.add_argument('-d', '--dataset_folder', type=str, required=True, help="/path/to/your/dataset")
    parser.add_argument('-e', '--model_path', type=str, required=True)
    parser.add_argument('-p', '--prompt_path', type=str, required=True)
    parser.add_argument('-f', '--offset', type=int, default=0)
    args = parser.parse_args()
    main(args)
