python inference.py \
      --num_layers=48 --latent_dim=48 --num_gpu=4 \
      --output_folder="/l/users/gus.xia/fathinah/expotion/exp_0317/output" \
      --feat_paths="test.lst" \
      --model_path="/l/users/gus.xia/fathinah/expotion/exp_0317/raft_must_b10_e20_lr1e-02/diff_19_end.pth" \
      --prompt_path="demo/prompt.txt" \
      --offset=5
