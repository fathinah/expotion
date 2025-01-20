import subprocess
import argparse
import os 
import shutil
from preprocess.prep_dataset import *

def execute_scripts(f):
    """
    Execute a series of Python scripts with specified arguments.
    """
    try:
        meta_f = f.strip('/') + '_meta'
        os.mkdir(meta_f, exists_ok=True)
        vid_f = os.path.join(meta_f, 'video') # tnj_0_meta/video
        # audio_f = os.path.join(meta_f, 'audio')
        # rvq_f = 
        os.mkdir(meta_f, exists_ok=True)

        # cleanname+split     
        subprocess.run(f"python preprocess/clean_name.py --directory {f}", check=True, shell=True)
        subprocess.run(f"python preprocess/splitter.py --folder {f}", check=True, shell=True)
        sep_f = f + '_separated' 
        shutil.move(sep_f, vid_f) 
        # resample
        subprocess.run(f"python preprocess/resample.py --input {vid_f}", check=True, shell=True)
        res_f = vid_f + '_resampled' # tnj_0_meta/video_resampled
        # divide to 10 folders
        subprocess.run(f"python preprocess/distribute_files.py --directory {res_f}", check=True, shell=True)
        # feat extraction (FE)
        subprocess.run(f"python preprocess/feature_extraction.py --main {res_f} --sub folder_1 folder_2 folder_3 folder_4 folder_5 folder_6 folder_7 folder_8 folder_9 folder_10 --gpus 0 0 0 0 0 1 1 1 1 1", check=True, shell=True)
        # extratc mp3 and rvq
        mp3_f = os.path.join(meta_f,'audio')
        rvq_f = os.path.join(meta_f,'rvq')
        os.mkdir(mp3_f, exists_ok=True)
        os.mkdir(rvq_f, exists_ok=True)
        
        output_file = f"data/train_vid+face_{os.dir.basename(f)}.lst"
        save_filenames_to_lst(folder_path, output_file, include_subfolders=True)

        
        # Add more scripts as needed
        print("All scripts executed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing: {e.cmd}")
        print(f"Return code: {e.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Python scripts in series.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder which contain raw videos.")
    args = parser.parse_args()

    execute_scripts(args.folder)
