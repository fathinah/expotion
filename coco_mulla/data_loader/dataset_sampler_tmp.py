import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *
import numpy as np
import torch

def load_data_from_path(path, idx, sec):
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    data_index = []
    ii = 0
    for i, line in enumerate(lines):
        line = line.rstrip()
        feat_path = line.split(" ")[0]
        onset = float(line.split(" ")[1])
        offset = float(line.split(" ")[2])
        face = torch.load(feat_path)['face_p'].detach().cpu().numpy()
        if face.shape[0]!=500:
            print(face.shape)
        else:
            data.append({
                "path": feat_path,
                "data": {
                    "video": None,
                    "face":  face,
                    "motion":None
                }
            })
            onset = math.ceil(onset)
            offset = int(offset)
            data_index += [[idx, ii, j] for j in range(onset, offset, 10)]
            ii+=1
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
            "tom and jerry sound with no noise"
        ]

    def get_prompt(self):
        return self.descs[self.rng.randint(len(self.descs))]

    def load_data(self, set_id, song_id):
        data = self.data[set_id]
        # Cache music, video, etc. once
        if "music" not in data[song_id]["data"]:
            # video  = data[song_id]["data"]["video"]
            # motion = data[song_id]["data"]["motion"]
            face   = data[song_id]["data"]["face"]

            music_path = data[song_id]["path"].replace('pth', 'npy').replace('face','rvq')
            music = np.load(music_path)
            result = {
                "music":  music,
                "video":  None,
                "motion": None,
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
        face   = data["face"]

        pad_frame = np.zeros((501 -  face.shape[0], face.shape[1]), dtype=face.dtype)
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
            "desc": desc,
            "face": face,
            "video": None,
            "motion":None
        }
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
    
    out['video'] = None
    out['motion'] = None 
    face_list = [torch.from_numpy(b["face"]) for b in batch]
    out["face"] = torch.stack(face_list, 0)

    return out
