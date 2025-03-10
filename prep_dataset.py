import os
from coco_mulla.utilities.encodec_utils import extract_rvq
import os
from moviepy.editor import VideoFileClip
import librosa
import torch
import numpy as np
from tqdm import tqdm

def extract_audio_from_mp4(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".mp4"):  # Check for MP4 files
            video_path = os.path.join(folder_path, file_name)
            output_audio_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")
            
            print(f"Processing: {file_name}")
            try:
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(output_audio_path, codec="libmp3lame")
                print(f"Audio extracted to: {output_audio_path}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")


def save_filenames_to_lst(folder_path, output_file, include_subfolders=True):
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            pass 

    file_names = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_names.append(os.path.join(root, file) if include_subfolders else file)
        
        if not include_subfolders:
            break
    
    with open(output_file, 'w') as lst_file:
        for file_name in file_names:
            lst_file.write(file_name + '\n')
    
    print(f"File names saved to {output_file}")

def compute_rvq(folder_path,output_path):
    sr=16000
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in tqdm(os.listdir(folder_path)):
        i_ = os.path.join(folder_path, i)
        wav, _ = librosa.load(i_, sr=sr, mono=True)
        wav = torch.from_numpy(wav).to(device)[None, None, ...]
        mix_rvq = extract_rvq(wav, sr=sr)
        np.save(os.path.join(output_path,i.replace(".mp3",".npy")), mix_rvq.cpu().numpy())

def remove_non_mp3_files(folder_path):
    """
    Remove non-MP3 files from a folder.

    Parameters:
        folder_path (str): Path to the folder.
    """
    # Get a list of all files in the folder
    for file_name in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a file (not a folder) and its extension is not .mp3
        if os.path.isfile(file_path) and not file_name.lower().endswith('.mp3'):
            try:
                os.remove(file_path)  # Remove the non-MP3 file
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

import os
import shutil

def release_files_to_folder(parent_folder):
    """
    Move all files from subfolders to the parent folder.
    """
    try:
        for root, dirs, files in os.walk(parent_folder):
            # Skip the parent folder itself
            if root == parent_folder:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                new_path = os.path.join(parent_folder, file)
                
                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    print(f"File already exists in the parent folder: {new_path}")
                    continue
                
                # Move the file to the parent folder
                shutil.move(file_path, new_path)
                print(f"Moved: {file_path} -> {new_path}")
        print("All files have been moved to the parent folder.")

    except Exception as e:
        print(f"An error occurred: {e}")


    

root_path = "/home/coder/laopo/data/dataset_tomnjerry/filtered/tnj_4"
face_folder = os.path.join(root_path,'face')
# release_files_to_folder(face_folder)
folder_path = os.path.join(root_path,'video')  # Replace with the path to your folder
output_file = "data/train_vid+face_4.lst"  # Replace with your desired output file name
mp3_folder = os.path.join(root_path,'audio')
rvq_path = os.path.join(root_path,'rvq')
save_filenames_to_lst(folder_path, output_file, include_subfolders=True)
extract_audio_from_mp4(folder_path, mp3_folder)
remove_non_mp3_files(mp3_folder)
compute_rvq(mp3_folder,rvq_path)
print("done!")
