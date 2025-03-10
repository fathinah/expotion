CUDA_VISIBLE_DEVICES=0,1 python inference.py \
      --num_layers=48 --latent_dim=12 \
      --output_folder="demo/output/video_0107" \
      --video_path="/home/coder/laopo/demo_video/demo_humanface.mp4" \
      --model_path="/home/coder/laopo/expotion/exp_0114/face_must_b20_e20_lr1e-02/diff_11_end.pth" \
      --prompt_path="demo/input/let_it_be/let_it_be.prompt.txt" \
      --offset=5
