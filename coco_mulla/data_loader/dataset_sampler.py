import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *
import numpy as np
from scipy.interpolate import CubicSpline
import torch

def interpolate(data):
    dim = data.shape[1]
    data_new = np.zeros((500, dim))
    seq_len = data.shape[0]
    t_original = np.linspace(0, 1, seq_len)
    t_new = np.linspace(0, 1, 500)
    for i in range(dim):
        cs = CubicSpline(t_original, data[:, i])
        data_new[:, i] = cs(t_new)
    return data_new

def replicate_frames(data, factor=10):
    """
    Replicate each frame 'factor' times along the time axis.
    E.g. a (50, dim) array -> (500, dim) array if factor=10.
    """
    return np.repeat(data, repeats=factor, axis=0)



def load_data_from_path(path, idx, sec):
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    data_index = []
    for i, line in enumerate(lines):
        line = line.rstrip()
        f_path = line.split(" ")[0]
        onset = float(line.split(" ")[1])
        offset = float(line.split(" ")[2])
        video = torch.load(os.path.join(f_path, "internvid.mp4.pt")).to(torch.float32).detach().cpu().numpy()
        motion = np.load(os.path.join(f_path, "motion.npy"))[:, 0, :]
        face = np.load(os.path.join(f_path, "face.npy"))


        # Replace spline interpolation with replication
        video = replicate_frames(video, 10)   # shape (500, D_video)
        motion = replicate_frames(motion, 10) # shape (500, D_motion)
        face = replicate_frames(face, 10)     # shape (500, D_face)


        data.append({
            "path": f_path,
            "data": {
                "video": interpolate(video),
                "face":  interpolate(face),
                "motion":interpolate(motion)
            }
        })
        onset = math.ceil(onset)
        offset = int(offset)
        data_index += [[idx, i, j] for j in range(onset, offset, 10)]
    return data, data_index

class Dataset(BaseDataset):
    def __init__(self, 
                 path_lst, 
                 cfg, 
                 rid, 
                 sampling_prob=None, 
                 sampling_strategy=None, 
                 inference=False,
                 is_video=True,     # <-- NEW ARGS
                 is_motion=True,    # <-- NEW ARGS
                 is_face=True):     # <-- NEW ARGS
        super(Dataset, self).__init__()
        self.rid = rid
        self.rng = np.random.RandomState(42 + rid * 100)
        self.cfg = cfg
        self.data = []
        self.data_index = []
        
        # NEW: store booleans
        self.is_video  = is_video
        self.is_motion = is_motion
        self.is_face   = is_face

        for i, path in enumerate(path_lst):
            data, data_index = load_data_from_path(path, i, cfg.sample_sec)
            self.data.append(data)
            self.data_index += data_index

        self.f_len = len(self.data_index)
        print("num of files", self.f_len)

        self.epoch = 0
        self.f_offset = 0
        self.inference = inference

        self.descs = [
            "A realistic and high quality soundtrack and sound effect for the video",
            "High quality sountrack for a cartoon movie",
            "A realistic and high quality soundtrack and sound effect for the video",
            "High quality sountrack for a cartoon movie"
        ]

    def get_prompt(self):
        return self.descs[self.rng.randint(len(self.descs))]

    def load_data(self, set_id, song_id):
        data = self.data[set_id]
        # Cache music, video, etc. once
        if "music" not in data[song_id]["data"]:
            video  = data[song_id]["data"]["video"]
            motion = data[song_id]["data"]["motion"]
            face   = data[song_id]["data"]["face"]

            music_path = os.path.join(data[song_id]["path"], "music.npy")
            music = np.load(music_path)
            result = {
                "music":  music,
                "video":  video,
                "motion": motion,
                "face":   face
            }
            data[song_id]["data"] = result
        return data[song_id]["data"]

    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        set_id, sid, sec_id = self.data_index[idx]
        data = self.load_data(set_id, sid)

        # Extract everything
        music  = data["music"]
        video  = data["video"]
        motion = data["motion"]
        face   = data["face"]

        # Pad the last frame to match your existing code
        pad_frame = np.zeros((1, video.shape[1]), dtype=video.dtype)
        video = np.concatenate([video, pad_frame], axis=0)

        pad_frame = np.zeros((1, motion.shape[1]), dtype=motion.dtype)
        motion = np.concatenate([motion, pad_frame], axis=0)

        pad_frame = np.zeros((1, face.shape[1]), dtype=face.dtype)
        face = np.concatenate([face, pad_frame], axis=0)

        cfg = self.cfg
        st = sec_id
        ed = st + cfg.sample_sec
        frame_st = int(st * cfg.frame_res)
        frame_ed = int(ed * cfg.frame_res)

        # Slice music frames
        music = music[:, frame_st: frame_ed]

        desc = self.get_prompt()
        
        # Build the output dict dynamically 
        out = {
            "music": music,
            "desc": desc
        }
        if self.is_video:
            out["video"] = video
        if self.is_motion:
            out["motion"] = motion
        if self.is_face:
            out["face"] = face

        return out

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e
        self.rng.shuffle(self.data_index)

def collate_fn(batch):
    """
    Collect only the keys that appear in the first sample 
    (since each sample in the batch must have the same keys).
    """
    out = {}
    
    # 'music' and 'desc' are always used
    music_list = [torch.from_numpy(b["music"]) for b in batch]
    out["music"] = torch.stack(music_list, 0)
    out["desc"]  = [b["desc"] for b in batch]

    # check if 'video' is in the first item
    if "video" in batch[0]:
        video_list = [torch.from_numpy(b["video"]) for b in batch]
        out["video"] = torch.stack(video_list, 0)

    # check if 'motion' is in the first item
    if "motion" in batch[0]:
        motion_list = [torch.from_numpy(b["motion"]) for b in batch]
        out["motion"] = torch.stack(motion_list, 0)

    # check if 'face' is in the first item
    if "face" in batch[0]:
        face_list = [torch.from_numpy(b["face"]) for b in batch]
        out["face"] = torch.stack(face_list, 0)

    return out
