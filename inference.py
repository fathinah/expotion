import argparse
import librosa
import shutil
# from marlin_pytorch import Marlin
from coco_mulla.models.model_motion import CoCoMulla
from coco_mulla.utilities import *
import cv2
from config import TrainCfg
import torch.nn.functional as F
from moviepy.editor import VideoFileClip, AudioFileClip
import subprocess
import shlex
import soundfile as sf
import torch.nn.functional as F
import multiprocessing as mp
import ast

def motion_interpolate(v):
    v = v.unsqueeze(0).permute(0, 2, 1) 
    v_interpolated = F.interpolate(v, size=500, mode='linear', align_corners=True)  # Shape: [1, 256, 500]
    v_interpolated = v_interpolated.permute(0, 2, 1).squeeze(0)  # Shape: [500, 256]
    return v_interpolated

def resample(input, output, rate=5*16):
    print(output)
    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps',fps)
    command = f"ffmpeg -i {input} -vf fps={rate} -y {output}" 
    command_list = shlex.split(command)
    subprocess.run(command_list)

def compute_feature_face(model, video):
    features = model.extract_video(video,sample_rate = 1, stride = 16,crop_face=True,detector_device=DEVICE)
    print(features.shape)
    return features

def generate(model, batch):
    with torch.no_grad():
        gen_tokens = model(**batch)
        gen_audio = model.musicgen.compression_model.decode(gen_tokens, None)
    return gen_tokens, gen_audio


# def generate_mask(xlen):
#     names = ["face", "face-body", "face-vid", "face-body-vid"]
#     mask = torch.zeros([4, 2, xlen]).to(device)
#     mask[1, 1] = 1
#     mask[2, 0] = 1
#     mask[3] += 1
#     return mask, names

def generate_mask(xlen, device):
    names = ["face"]
    mask = torch.zeros([1, 2, xlen]).to(device)
    return mask, names

def load_data(feat_path, output_folder, offset, device):
    sr = TrainCfg.sample_rate ## 16000
    res = TrainCfg.frame_res ## 25
    sample_sec = TrainCfg.sample_sec ##10
    # video_name = video_path[:-4].split('/')[-1]
    # vp_resampled = output_folder+ f"/{video_name}_resampled.mp4"
    # resample(video_path, vp_resampled)
    # face_model = Marlin.from_online("marlin_vit_base_ytf").to(DEVICE)
    # face = compute_feature_face(face_model,vp_resampled)
    # T, _ = face.shape
    # face_p = face.permute(1, 0).unsqueeze(0)
    # new_T = T*10
    # face_p = F.interpolate(face_p, size=new_T, mode='linear', align_corners=True)
    # face_p = face_p.squeeze(0).permute(1, 0)
    # face_p = face_p.cpu().numpy()
    face_p = motion_interpolate(torch.load(feat_path)).detach().numpy() ##change
    # face_p = torch.load(feat_path)['face_p'].cpu().numpy() ##chnage


    new_T = face_p.shape[0]
    print('new_T',new_T)

    vid = np.zeros((new_T,1))
    body = np.zeros((new_T,1))
    print('vid, body, face', vid.shape, body.shape, face_p.shape)

    face = crop(face_p[None, ...], sample_sec, res,offset=offset)
    body = crop(body[None, ...], sample_sec, res, offset=offset)
    vid = crop(vid[None, ...],sample_sec, res, offset=offset)
    print("face",face.shape)
    print("vid",vid.shape)
    print("body",body.shape)
    face = torch.from_numpy(face).to(device).float()
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
    print(st,ed)
    print(x.shape[1])
    assert x.shape[1] > st
    return x[:, st: ed] ## 251 seq length

def replace_audio_in_video(video_path, audio_path, output_path):
    """
    Replace the audio of a video with a new audio file.
    
    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to the new audio WAV file.
        output_path (str): Path to the output video file.
    """
    # Load video and audio
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration
    new_audio = AudioFileClip(audio_path)
    audio_duration = new_audio.duration
    duration = min(video_duration,audio_duration)
    print('duration',duration)
    print('vfps',video_clip.fps)
    print('afps', new_audio.fps)
    new_video = video_clip.subclip(0, duration)
    new_audio = new_audio.subclip(0, duration)
    

    # Replace audio
    video_with_new_audio = new_video.set_audio(new_audio)
    
    # Write the output video
    video_with_new_audio.write_videofile(output_path, codec="libx264", audio_codec="aac",fps=30)

def save_pred(feat_path, output_folder, wav, tags, pred):
    video_path = feat_path.replace("raft_5fps","clips").replace(".pt",".mp4") ##change
    video_name = video_path[:-4].split('/')[-1]
    mkdir(output_folder)
    output_folder = output_folder+'/'+video_name
    mkdir(output_folder) ## change
    shutil.copy(video_path, output_folder)

    for i in [0]: ## change
        audio_p = os.path.join(output_folder, f'{tags[i]}.wav')
        wav_ = wav[i].squeeze().cpu()
        sf.write(audio_p, wav_, 32000)
        print(output_folder+f"/{video_name}_{tags[i]}.mp4")
        replace_audio_in_video(video_path, audio_p,output_folder+f"/{video_name}_{tags[i]}.mp4")


def wrap_batch(face, body, vid, cond_mask, prompt):
    num_samples = len(cond_mask)
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
        # "max_n_frames": 500,
        "mode": "inference",
    }
    return batch


        
        
           
def process_videos_in_folder(lines, args,device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = CoCoMulla(TrainCfg.sample_sec,
                      num_layers=args.num_layers,
                      latent_dim=args.latent_dim).to(device)
    model.load_weights(args.model_path)
    model.eval()
    
    for line in lines:
            feat_path = line[0].split(' ')[0].rstrip('\n')
            caption = line[1]
            print('caption', caption)
            face, body, vid = load_data(feat_path=feat_path,
                                        output_folder=args.output_folder,
                                        offset=args.offset, device=device)
            cond_mask, names = generate_mask(face.shape[-2], device)
            batch = wrap_batch(face.to(device), body.to(device), vid.to(device), cond_mask.to(device), caption)
            print(batch['face'].shape,batch['body'].shape,batch['vid'].shape, batch['cond_mask'].shape, batch['desc'])
            pred, wav = generate(model=model,
                            batch=batch)
            save_pred(feat_path = feat_path, output_folder=args.output_folder,wav = wav,
                        tags=names,
                        pred=pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-n', '--num_layers', type=int)
    parser.add_argument('-l', '--latent_dim', type=int)
    parser.add_argument('-g', '--num_gpu', type=int,default=0)
    parser.add_argument('-v', '--feat_paths', type=str, default=None)
    parser.add_argument('-e', '--model_path', type=str)
    parser.add_argument('-p', '--prompt_path', type=str)
    parser.add_argument('-f', '--offset', type=int)
    mp.set_start_method('spawn', force=True)

    args = parser.parse_args()
    num_gpus = args.num_gpu
    new_files = []


    with open(args.feat_paths, "r") as f:
        video_files = f.readlines()

    for line in video_files:
        feat_path = line.split('|')[0].rstrip('\n')
        feat_name =  feat_path[:-3].split('/')[-1]
        caption = ast.literal_eval(line.split('|')[-1])
        caption = caption["caption"].replace("<s>","").replace("</s>","").replace("\n","").strip()
        if not os.path.exists(os.path.join(args.output_folder,feat_name)):
        # if len(os.listdir(os.path.join(args.output_folder,feat_name)))!=3:
            new_files.append([feat_path, caption])
    print('len',len(new_files))


    
    video_chunks = [new_files[i::num_gpus] for i in range(num_gpus)]
    processes = []
    for device_id, videos in enumerate(video_chunks):
        p = mp.Process(
            target=process_videos_in_folder,
            args=(videos, args,device_id)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
