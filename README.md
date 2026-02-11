# RADAR: Revealing Asymmetric Development of Abilities in MLLM Pre-training

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

This repository provides the official PyTorch implementation of the following paper: 
> [**RADAR: Revealing Asymmetric Development of Abilities in MLLM Pre-training**] <br>

## 🎯 News

**[2026.2.10]** 🚀 This project page has been built!

## 👨‍💻 Todo

- [ ] Release the M$^3$-Bench dataset
- [ ] Release the evaluation code of RADAR


## ⭐️ TL;DR
### 1. Installation
If you want to use our codebase for reproduction, you are recommended to build a new environment though the steps below. 

We take LLaVA-OneVision as an example:
(The following steps are just listed for Linux. If you are using macOS or Windows, please refer to [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main?tab=readme-ov-file#installation))
1. Clone this repository and navigate to RADAR folder
```
git clone https://github.com/Nieysh/RADAR.git
cd RADAR
```
2. Install Package
```
conda create -n llava python=3.10 -y
conda activate llava
python -m pip install --upgrade pip  # enable PEP 660 support
cd LLaVA-NeXT
python -m pip install -e .
pip install git+https://github.com/huggingface/transformers@745bbfe4bb2b61491dedd56e1e8ee4af8ef1a9ec
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

To evaluate other MLLMs for reproduction, please refer to [Qwen2-VL](https://github.com/xwjim/Qwen2-VL?tab=readme-ov-file#quickstart), [InternVL-3.5](https://internvl.readthedocs.io/en/latest/get_started/installation.html) for environment installation.

### 2. Data Preparation
Please follow the instructions below to prepare the checkpoints and data in directories:

(Take LLaVA-OneVision-0.5B (projector) as an example)
1. Download pretrained LLaVA-OneVision weight from [here](https://huggingface.co/lmms-lab/llava-onevision-projectors/tree/main).
2. Download Qwen2-0.5B-Instruct model weight from [here](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).
3. Download SigLIP-so400m-patch14-384 model weight from [here](https://huggingface.co/google/siglip-so400m-patch14-384).

For other models, refer to their huggingface respositories for downloading the pretrained weights:

| Model                      | HF Link                                                                |
|----------------------------|------------------------------------------------------------------------|
| InternVL3.5-1B-Pretrained  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Pretrained)  |
| InternVL3.5-2B-Pretrained  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Pretrained)  |
| InternVL3.5-4B-Pretrained  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Pretrained)  |
| InternVL3.5-8B-Pretrained  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Pretrained)  |
| InternVL3.5-14B-Pretrained | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B-Pretrained) |
| Qwen2-VL-2B                | [🤗 link](https://huggingface.co/Qwen/Qwen2-VL-2B)                     |
| Qwen2-VL-7B                | [🤗 link](https://huggingface.co/Qwen/Qwen2-VL-7B)                     |


4. Download M$^3$-Bench dataset from [here](xx) and unzip it to ```YOUR/PATH/TO/M3-BENCH/DATA```.

### Referenced Data Directory
```graphql
YOUR/PATH/TO/M3-BENCH/DATA
├─ general_visual_question_answering
|   ├─ images
|   ├─ MMBench_action_recognition_54_pretrained_mllm_eval.json
|   ├─ MMBench_attribute_comparison_44_pretrained_mllm_eval.json
|   ├─ .json data files of other tasks
├─ mathematical_reasoning
|   ├─ images
|   ├─ MathVista_1000_pretrained_mllm_eval.json
├─ Wiki_animal_identification
|   ├─ images
|   ├─ Wiki_animal_identification_2000_pretrained_mllm_eval.json
```



### 3. Evaluation

To reproduce the RADAR implementation on this codebase, you can follow these steps:
1. Specify the ```data_dir```, ```dataset``` and ```model_path``` in the script for RADAR calculation. (Also specify ```model_base``` for LLaVA-OneVision projectors)
2. Run the script to conduct RADAR evaluation for different models.

For LLaVA-OneVision (projectors):
```graphql
bash radar_eval_llava_ov.sh
```

For Qwen2-VL:
```graphql
bash radar_eval_qwen2l.sh
```

For InternVL-3.5:
```graphql
bash radar_eval_internvl3_5.sh
```

## Acknowledgement
This repo is based on the codebase of [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen2-VL](https://github.com/xwjim/Qwen2-VL?tab=readme-ov-file#quickstart), and [InternVL-3.5](https://internvl.readthedocs.io/en/latest/get_started/installation.html). Thanks for their impressive works!


[//]: # (## Citation)

[//]: # (If you find this work useful for your research, please cite our paper:)

[//]: # (```)

[//]: # (@article{huang2024deciphering,)

[//]: # (  title={Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate},)

[//]: # (  author={Huang, Qidong and Dong, Xiaoyi and Zhang, Pan and Zang, Yuhang and Cao, Yuhang and Wang, Jiaqi and Lin, Dahua and Zhang, Weiming and Yu, Nenghai},)

[//]: # (  journal={arXiv preprint arXiv:2410.07167},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (```)


