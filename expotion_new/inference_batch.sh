CUDA_VISIBLE_DEVICES=0,1 python inference_face_only_batch.py \
      --num_layers=48 --latent_dim=12 \
      --output_folder="demo/output/result_0321" \
      --video_path="data/test_eval.lst" \
      --dataset_folder="/home/coder/laopo/data" \
      --model_path="/home/coder/laopo/expotion/expotion_new/exp_0317/face_only_captions_human_b10_e50_lr1e-02/diff_49_end.pth" \
      --prompt_path="/home/coder/laopo/data/human-music-moves/v1/captions.json" \
      --offset=0