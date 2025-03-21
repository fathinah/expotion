import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *
import numpy as np
import random
import torch.nn.functional as F
import json
import ast

def video_interpolate(x):
    x = x.mean(dim=(-1, -2))  
    x_interpolated = F.interpolate(x, size=500, mode='linear', align_corners=False)
    x_interpolated = x_interpolated.transpose(1, 2).squeeze(0) 
    pad_frame = np.ones((1, x_interpolated.shape[1]))
    x_interpolated = np.concatenate([x_interpolated, pad_frame], axis=0)
    return x_interpolated

def motion_interpolate(v):
    v = v.unsqueeze(0).permute(0, 2, 1)
    v_interpolated = F.interpolate(v, size=500, mode='linear', align_corners=True) 
    v_interpolated = v_interpolated.permute(0, 2, 1).squeeze(0) 
    return v_interpolated



def load_data_from_path(path):
    lst = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        lst.append(line.rstrip('\n'))
    return lst


class Dataset(BaseDataset):
    def __init__(self, path_lst, cfg, rid, prompt_file, sampling_prob=None, sampling_strategy=None, inference=False):
        super(Dataset, self).__init__()
        self.rid = rid
        self.rng = np.random.RandomState(42 + rid * 100)
        self.cfg = cfg
        self.data = []
        self.data_index = []
        self.not_found = set()

        for i, path in enumerate(path_lst):
            data = load_data_from_path(path)
            self.data += data
        self.f_len = len(self.data)
        print("num of files", self.f_len)

        self.epoch = 0
        self.f_offset = 0
        self.inference = inference

        self.descs = []
        with open(prompt_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.descs.append(line.rstrip('\n'))
        print('desc',self.descs)   
        
        self.sampling_strategy = sampling_strategy
        if sampling_prob is None:
            sampling_prob = [0., 0.8]
        self.sampling_prob = sampling_prob
        print("samling strategy", self.sampling_strategy, sampling_prob)

    def get_prompt(self):

        prompts = self.descs
        return prompts[self.rng.randint(len(prompts))]

    def load_data(self, idx):
        tmp = 0
        while True:
            feature_id = idx if tmp==0 else random.sample(self.data,1)[0]##change 
            feats = feature_id.split('|')
            mix_id = feats[-2]
            feature_id = feats[0]
            caption = feats[-1]
            caption = ast.literal_eval(caption)
            caption = caption["caption"].replace("<s>","").replace("</s>","").replace("\n","").strip()
            # mix_id = os.path.join(os.path.dirname(feature_id).replace("raft","rvq"),os.path.basename(feature_id)).replace(".pt",".npy").strip() ##change
            # feature_id = os.path.join(os.path.dirname(feature_id).replace("video","face"),os.path.basename(feature_id)).replace(".mp4",".pth").strip() ##change
            if os.path.exists(mix_id) and os.path.exists(feature_id):
                break
            else:
                self.not_found.add(feature_id)
                tmp=1
        mix = np.load(mix_id)
        # resnet = video_interpolate(torch.load(feature_id))  #change
        # face = torch.load(feature_id)['face_p'] 
        motion = motion_interpolate(torch.load(feature_id))
        # video = torch.load(vid_id)

        motion = motion.cpu().detach().numpy() 
        result = {
            "mix": mix,
            "face": motion, ##change,
            "caption": caption
        }
        return result

    def track_based_sampling(self, seg_len):
        n = 2
        cond_mask = np.ones([n, seg_len])
        r = self.rng.randint(0, 4)
        if r == 0:
            cond_mask = cond_mask *0
        elif r == 1:
            cond_mask[0] = 0
        elif r == 2:
            cond_mask[1] = 0
        else:
            assert r == 3
        return cond_mask


    def prob_based_sampling(self, seg_len, sampling_prob):
        n = 2
        cond_mask = np.ones([n, seg_len])
        r = self.rng.rand()
        if r < sampling_prob[0]:
            cond_mask = cond_mask * 0.
        else:
            r = self.rng.rand(n)
            for i in range(n):
                if r[i] < sampling_prob[1]:
                    cond_mask[i] = 0
        return cond_mask

    def sample_mask(self, seg_len):
        if self.sampling_strategy == "track-based":
            return self.track_based_sampling(seg_len)
        if self.sampling_strategy == "prob-based":
            return self.prob_based_sampling(seg_len, self.sampling_prob)

    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        vid_id = self.data[idx]
        data = self.load_data(vid_id)
        mix = data["mix"]
        face = data["face"]
        caption = data["caption"]
        

        cfg = self.cfg
        st = 0
        ed = st + cfg.sample_sec
        frame_st = int(st * cfg.frame_res)
        frame_ed = int(ed * cfg.frame_res)

        # Calculate required lengths
        required_mix_length = frame_ed - frame_st
        required_face_length = frame_ed - frame_st + 1

        # Slice and pad 'mix'
        current_mix_length = mix.shape[1]  # Assuming 'mix' is a 2D array
        if current_mix_length < required_mix_length:
            padding_length = required_mix_length - current_mix_length
            mix = np.pad(mix, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
        else:
            mix = mix[:, frame_st: frame_ed]

        # Slice and pad 'face' along the first dimension
        current_face_length = face.shape[0]  # Assuming 'face' is a 2D array
        if current_face_length < required_face_length:
            padding_length = required_face_length - current_face_length
            face = np.pad(face, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
        else:
            face = face[frame_st: frame_ed + 1, :]

        T,_ = face.shape ##change
        seg_len = frame_ed - frame_st + 1
        cond_mask = self.sample_mask(seg_len)
        return {
            "mix": mix,
            "face":face,
            "vid": np.zeros((T,1)),
            "body": np.zeros((T,1)),
            "cond_mask":cond_mask,
            "desc": caption
        }

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e
        self.rng.shuffle(self.data)


def collate_fn(batch):
    mix = torch.stack([torch.from_numpy(d["mix"]) for d in batch], 0)
    face = torch.stack([torch.from_numpy(d["face"]) for d in batch], 0)
    vid = torch.stack([torch.from_numpy(d["vid"]) for d in batch], 0)
    body = torch.stack([torch.from_numpy(d["body"]) for d in batch], 0)
    cond_mask = torch.stack([torch.from_numpy(d["cond_mask"]) for d in batch], 0)
    desc = [d["desc"] for d in batch]
    return {
        "mix": mix,
        "face": face,
        "vid": vid,
        "body": body,
        "cond_mask": cond_mask,
        "desc": desc,
    }