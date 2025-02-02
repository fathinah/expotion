
## Quick Start

### Requirements
The project requires Python 3.11. See `requirements.txt`.

pip install -r requirements.txt

### Data Prep

1. Encode Video : expotion_dataset/video-enc/InternVideo/InternVideo2/multi_modality/internvid_inference.py
2. Encode Music : coco-mulla-repo/audio_enc.ipynb
3. Encode Motion : ## TBA @ CIAI
4. Encode Face : MARLIN/run.py & MARLIN/run2.py


### Training
Restrtucture folders at /home/fathinah.izzati/TJ/moved.py & moved2.py
Prepare shiver.lst 
python train.py -l shiver.lst -o expe_v2_2 --is-motion --is-video --is-face

### Inference

For individual file
python inference.py \
--num_layers=48 --latent_dim=48 \
--output_folder "/l/users/fathinah.izzati/coco-mulla-repo/expe_v2_2/videos" \
--model_path "/l/users/fathinah.izzati/coco-mulla-repo/expe_v2/diff_10_end.pth" \
--prompt_path "/l/users/fathinah.izzati/coco-mulla-repo/prompt.txt" \
--face_path "/home/fathinah.izzati/TJ/filtered/tnj_0/face/CopyofCopyof007-TheBowlingAlleyCat_clip_015_140-150.pth" \
--motion_path "/home/fathinah.izzati/TJ/filtered/Full_Tom_and_Jerry_Shiver_Me_Whiskers_2006_HD/chunk_327/motion.npy" \
--video_path "/home/fathinah.izzati/TJ/filtered/Full_Tom_and_Jerry_Shiver_Me_Whiskers_2006_HD/chunk_327/internvid.mp4.pt" \


python inference_bulk.py \
--num_layers=48 --latent_dim=48 \
--output_folder "/l/users/fathinah.izzati/coco-mulla-repo/expe_v2/generated_caption1" \
--model_path "/l/users/fathinah.izzati/coco-mulla-repo/expe_1/models/diff_3_1.pth" \
--prompt_path "/l/users/fathinah.izzati/coco-mulla-repo/prompt.txt" \
--chunks_folder "/home/fathinah.izzati/TJ/filtered/tnj_0/face"
--is-face
### to do
1. Bulk inference
2. Training for real using shiver me (different combination controls)
3. Interpolation change experiment
4. Motion embedding fix! Pake optical flow.
5. Add to powerpoint
