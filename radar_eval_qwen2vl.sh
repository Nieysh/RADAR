CUDA_VISIBLE_DEVICES=0 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset MMBench \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=1 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset MMMU_Pro \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=2 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset MathVista \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=3 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset SeePhys \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=4 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset Wiki_animal \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=5 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset Wiki_plant \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=6 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset Wiki_celebrity \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=7 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset Wiki_attraction \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs &

CUDA_VISIBLE_DEVICES=0 python Qwen2VL/qwen2vl_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/Qwen2-VL-2B \
--dataset Spatial_reasoning \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs

