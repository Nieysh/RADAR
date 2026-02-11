CUDA_VISIBLE_DEVICES=0 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
--vision_tower /YOUR/PATH/TO/siglip-so400m-patch14-384 \
--dataset MMBench \
--data_dir /YOUR/PATH/TO/DATA \
--output_dir ./outputs

#CUDA_VISIBLE_DEVICES=1 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset MMMU_Pro \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=2 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset MathVista \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=3 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset SeePhys \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=4 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset Wiki_animal \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=5 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset Wiki_plant \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=6 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset Wiki_celebrity \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=7 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset Wiki_attraction \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs &
#
#CUDA_VISIBLE_DEVICES=0 python LLaVA-NeXT/llava-ov_pretrained_mllm_eval.py \
#--model_path /YOUR/PATH/TO/llava-onevision-projectors/0.5b \
#--model_base /YOUR/PATH/TO/Qwen/Qwen2-0.5B-Instruct \
#--dataset Spatial_reasoning \
#--data_dir /YOUR/PATH/TO/DATA \
#--output_dir ./outputs

