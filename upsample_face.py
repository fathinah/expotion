import torch
from transformers import T5EncoderModel, T5Tokenizer 

# import numpy as np
# import cv2
# import os

# def get_video_length(video_path):
#     """
#     Get the duration of a video in seconds using OpenCV.

#     Args:
#         video_path (str): Path to the video file.

#     Returns:
#         float: Length of the video in seconds.
#     """
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("Could not open the video file.")
        
#         fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
#         frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
        
#         if fps > 0:
#             duration = frame_count / fps
#         else:
#             raise ValueError("FPS is zero, invalid video file.")
        
#         cap.release()
#         return duration
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# def read_lst_file(file_path):
#     """
#     Reads lines from a .lst file.

#     Args:
#         file_path (str): Path to the .lst file.

#     Returns:
#         list: List of lines from the file.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             lines = [line.strip() for line in file.readlines()]
#         return lines
#     except FileNotFoundError:
#         print(f"Error: File not found at {file_path}")
#         return []
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return []

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Read lines from a .lst file.")
#     parser.add_argument("--file", type=str, required=True, help="Path to the .lst file.")
#     args = parser.parse_args()
#     valid_lines = []
#     lines = read_lst_file(args.file)
#     print("Lines from the .lst file:")
#     for line in lines:
#         print(line)
#         video_path = line.strip('\n')
#         if not os.path.exists(video_path):
#             print(f"Warning: Video file does not exist: {video_path}")
#             continue
            
#         length = get_video_length(video_path)
#         if length is None or length >= 10:
#             valid_lines.append(line)  # Keep this line if video length is valid
#             print(f"adding {line}")
#         # Overwrite the .lst file with valid entries
#     with open(args.file, 'w') as file:
#         file.writelines(valid_lines)

#     print("Filtered .lst file successfully.")

# python upsample_face.py --file /home/coder/laopo/expotion/data/train_vid+face_2.lst

import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
folder = '/home/coder/laopo/data/dataset_tomnjerry/filtered/tnj_4/face'
for i in tqdm(os.listdir(folder)):
    pth_file = os.path.join(folder,i)
    face = torch.load(pth_file)['face']
    T, _ = face.shape
    # print(T)
    face_p = face.permute(1, 0).unsqueeze(0)
    new_T = T*10
    upsampled_tensor = F.interpolate(face_p, size=new_T, mode='linear', align_corners=True)
    upsampled_tensor = upsampled_tensor.squeeze(0).permute(1, 0)
    # print(upsampled_tensor.shape)  # Should be (T*10, features), e.g., (500, 3)
    torch.save({'face':face, 'face_p':upsampled_tensor},pth_file)
#     # break
# # path = "data/train_vid+face.lst"
# # with open(path, "r") as f:
# #     lines = f.readlines()
# # for line in lines:
# #     print(line)
# #     line = line.rstrip()
# #     print(line)
# #     break