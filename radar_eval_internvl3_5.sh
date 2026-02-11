CUDA_VISIBLE_DEVICES=0 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset MMBench \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=1 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset MMMU_Pro \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=2 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset MathVista \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=3 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset SeePhys \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=4 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset Wiki_animal \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=5 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset Wiki_plant \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=6 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset Wiki_celebrity \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=7 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset Wiki_attraction \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=0 python InternVL/intern3_5vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/InternVL3_5-1B-Pretrained \
--dataset Spatial_reasoning \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs

