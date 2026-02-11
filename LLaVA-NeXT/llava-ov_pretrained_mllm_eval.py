import ast
import os
import sys
from PIL import Image
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
import torch
import copy
import numpy as np
import random
from tqdm import tqdm
import requests
from io import BytesIO
import glob
import re
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--model_base", type=str, default=None)
parser.add_argument("--vision_tower", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default='.')
args = parser.parse_args()

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(0)

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def extract_images_from_text(text):
    return re.findall(r"<image \d+>", text)

def replace_image_with_image_token(text, images, img_token):
    for image in images:
        text = text.replace(image, f"{img_token}")
    return text

def get_image_files_from_all_images(img_dir, all_images, q_id):
    all_images_files = []
    for image in all_images:
        all_images_files.append(f"{img_dir}/{q_id}_{image.strip('<').strip('>').replace(' ', '_')}.png")
    return all_images_files

##########  model #############
model_path = args.model_path
model_base = args.model_base

dtype = torch.float16
model_name = "llava_qwen"
device = "cuda:0"
device_map = "auto"
llava_model_args = {
        "multimodal": True,
    }
conv_template = "plain"
max_token_per_img = 729
overwrite_config = {"vision_tower": args.vision_tower, "mm_vision_tower": args.vision_tower}
llava_model_args["overwrite_config"] = overwrite_config

tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, device_map=device_map, attn_implementation='eager',**llava_model_args)
model.eval()
##########  model #############


if 'MMBench' in args.dataset:
    ## MMBench
    dir_name = "general_visual_question_answering"

if 'MMMU_Pro' in args.dataset:
    ## MMMU_Pro
    dir_name = "multiple_discipline_reasoning"

if 'MathVista' in args.dataset:
    ## MathVista
    dir_name = "mathematical_reasoning"

if 'SeePhys' in args.dataset:
    ## SeePhys
    dir_name='physical_reasoning'

if 'Wiki_animal' in args.dataset:
    ## Wiki_animal
    dir_name = "Wiki_animal_identification"

if 'Wiki_plant' in args.dataset:
    ## Wiki_plant
    dir_name = "Wiki_plant_identification"

if 'Wiki_attraction' in args.dataset:
    ## Wiki_attraction
    dir_name = "Wiki_attraction_identification"

if 'Wiki_celebrity' in args.dataset:
    ## Wiki_celebrity
    dir_name = "Wiki_celebrity_identification"

if 'Spatial_reasoning' in args.dataset:
    dir_name = "spatial_reasoning"

pretrain_benchmark_dir = os.path.join(args.data_dir, dir_name)
pretrain_benchmark_img_dir = os.path.join(args.data_dir, dir_name, 'images')
pretrain_benchmark_files = glob.glob(os.path.join(pretrain_benchmark_dir, "*.json"))
save_dir = os.path.join(args.output_dir, dir_name)

all_cat_res = {}

for pretrain_benchmark_file in tqdm(pretrain_benchmark_files):
    data = load_json(pretrain_benchmark_file)
    q_ids = list(data.keys())

    total_success = 0
    all_res = []
    all_avg_prob = []

    save_file_dir = os.path.join(save_dir, pretrain_benchmark_file.split("/")[-1].split(".json")[0])
    save_file = '_'.join(model_path.split('/')[-2:]) + f"_{max_token_per_img}" + '.json'
    print(save_file_dir + '/' + save_file)
    # if os.path.exists(save_file_dir + '/' + save_file):
    #     print(f"{save_file_dir + '/' + save_file} exists, skip!")
    #     continue

    for q_id in tqdm(q_ids):
        item = data[q_id]

        # category
        category = pretrain_benchmark_file.split("/")[-1].split(".json")[0]

        # image
        img_files = []
        if 'MMMU_Pro' in pretrain_benchmark_file:
            all_images = extract_images_from_text(data[q_id]["question"])
            img_cate = '_'.join(pretrain_benchmark_file.split('/')[-1].split('.')[0].split("_")[2:-4])
            img_files = get_image_files_from_all_images(pretrain_benchmark_img_dir + f"/{img_cate}", all_images, q_id)
        elif 'MMBench' in pretrain_benchmark_file:
            img_cate = '_'.join(pretrain_benchmark_file.split('/')[-1].split('.')[0].split("_")[1:-4])
            img_files = [f"{pretrain_benchmark_img_dir}/{img_cate}/{q_id}.jpg"]
        elif 'SeePhys' in pretrain_benchmark_file:
            for im_idx in range(item['img_num']):
                img_files.append(os.path.join(pretrain_benchmark_img_dir, f"{q_id.split('--')[0]}_{im_idx}.png"))
        else:
            img_files = [f"{pretrain_benchmark_img_dir}/{q_id.split('--')[0]}.jpg"]

        # question
        if 'MMMU_Pro' in pretrain_benchmark_file:
            question = replace_image_with_image_token(data[q_id]["question"], all_images, DEFAULT_IMAGE_TOKEN)
            if extract_images_from_text(data[q_id]["ori_options"]): # if images appear in options, then include options in question prompt
                distraction = item['distraction']
                if random.random() > 0.5:
                    options = [data[q_id]["answer"]] + distraction
                    for option in options:
                        img_files.extend(
                            get_image_files_from_all_images(pretrain_benchmark_img_dir + f"/{img_cate}",
                                                            extract_images_from_text(option), q_id))
                    options_letters = ["A", "B"]
                    options = [
                        f"{chr(ord('A') + i)}. {replace_image_with_image_token(option, extract_images_from_text(option), DEFAULT_IMAGE_TOKEN)}"
                        for i, option in enumerate(options)]
                else:
                    options = distraction + [data[q_id]["answer"]]
                    for option in options:
                        img_files.extend(
                            get_image_files_from_all_images(pretrain_benchmark_img_dir + f"/{img_cate}",
                                                            extract_images_from_text(option), q_id))
                    options_letters = ["B", "A"]
                    options = [
                        f"{chr(ord('A') + i)}. {replace_image_with_image_token(option, extract_images_from_text(option), DEFAULT_IMAGE_TOKEN)}"
                        for i, option in enumerate(options)]
                question = question + " " + ' '.join(options) + " Please select the correct option: "
        elif 'MMBench' in pretrain_benchmark_file or 'MathVista' in pretrain_benchmark_file:
            question = item['question']
            question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        elif 'SeePhys' in pretrain_benchmark_file:
            question = ' '.join([DEFAULT_IMAGE_TOKEN * item['img_num']]) + ' ' + item['question']
        else:
            question = item['question']
            question =  DEFAULT_IMAGE_TOKEN + ' ' + question

        # answer
        answers = []
        if 'MMMU_Pro' in pretrain_benchmark_file:
            if extract_images_from_text(data[q_id]["ori_options"]):
                answers = options_letters
            else:
                distraction = item['distraction']
                if len(distraction) > 1:
                    distraction = random.sample(distraction, 1)
                answers = [item["answer"]] + distraction
        else:
            answer = item['answer']
            distraction = item['distraction']
            answers = [answer] + distraction

        print(f"Q_ID: {q_id} Question: {question} Candidates: {answers}")

        if len(img_files) > 12:
            print(f"Too many images for {q_id}, skip!")
            continue
        images = []
        for img_file in img_files:
            images.append(Image.open(img_file).convert("RGB"))
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = [_image.to(dtype=dtype, device=device) for _image in image_tensors]
        image_sizes = [image.size for image in images]

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_prompt_woanswer_len = input_ids.size(1) + 729 * len(img_files) - len(img_files)  # -1 is because <image> is replaced with img tokens

        answers_avg_logits = []
        answers_ori_logits = []
        for answer in answers:
            input_ids_wanswer = (
                tokenizer_image_token(prompt + answer, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            answer_len = input_ids_wanswer.size(1) - input_ids.size(1)

            with torch.inference_mode():
                output = model(
                    input_ids_wanswer,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    return_dict=True,
                )
            answer_logits = []
            for i, logit in enumerate(output['logits']):
                x = 0
                for idx in range(input_prompt_woanswer_len - 1, logit.size(0) - 1):
                    logit_cur_tok = logit[idx, input_ids_wanswer[i, input_ids.size(1) + x]]
                    x += 1
                    answer_logits.append(logit_cur_tok.item())
                print(f"input_prompt_woanswer_len: {input_prompt_woanswer_len} logit size: {logit.size(0)}")
            answers_ori_logits.append(answer_logits.copy())
            answers_avg_logits.append(sum(answer_logits) / len(answer_logits))

        answer_prob = torch.softmax(torch.tensor(answers_avg_logits), dim=0)
        success = answer_prob.argmax().item() == 0
        if success:
            total_success += 1

        res = {
            "q_id": q_id,
            "question": question,
            "answers": answers,
            "success": success,
            "answer_prob": answer_prob.tolist(),
            "answers_avg_logits": answers_avg_logits,
            "answers_ori_logits": answers_ori_logits,
        }
        all_res.append(res)
        all_avg_prob.append(answer_prob[0].item())

        print(f"Success: {success}, answers_avg_logits: {answers_avg_logits}")

    accuracy = total_success / len(q_ids)
    avg_prob = sum(all_avg_prob) / len(all_avg_prob)
    file_result = f"Success Rate: {total_success}/{len(q_ids)}:{accuracy*100:.2f}%, Answer Prob: {avg_prob}\n"
    all_cat_res[category.split('_pretrained_mllm_eval')[0]] = {
        "accuracy": accuracy,
        "avg_prob": avg_prob
    }


    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir, exist_ok=True)
    with open(os.path.join(save_file_dir, save_file), 'w') as f:
        json.dump(all_res, f)

    save_txt_file = '_'.join(model_path.split('/')[-2:]) + f"_{max_token_per_img}" + '.txt'
    with open(os.path.join(save_file_dir, save_txt_file), 'w') as f:
        f.write(file_result)
    print(save_file_dir + '/' + save_txt_file)
    print(file_result)

if "MMBench" in pretrain_benchmark_file or "MMMU_Pro" in pretrain_benchmark_file:
    all_save_file_dir = os.path.join(save_dir, 'total')
    all_save_file = '_'.join(model_path.split('/')[-2:]) + f"_{max_token_per_img}" + '.json'
    print(all_save_file_dir + '/' + all_save_file)
    if not os.path.exists(all_save_file_dir):
        os.makedirs(all_save_file_dir, exist_ok=True)
    with open(os.path.join(all_save_file_dir, all_save_file), 'w') as f:
        json.dump(all_cat_res, f)

    all_avg_acc = sum(all_cat_res[k]['accuracy'] for k in all_cat_res.keys()) / len(all_cat_res)
    all_avg_prob = sum(all_cat_res[k]['avg_prob'] for k in all_cat_res.keys()) / len(all_cat_res)
    all_file_result = f"Average Success Rate: {sum(all_cat_res[k]['accuracy'] for k in all_cat_res.keys())}/{len(all_cat_res)}:{all_avg_acc*100:.2f}%, Average Answer Prob: {all_avg_prob}\n"
    all_save_txt_file = '_'.join(model_path.split('/')[-2:]) + f"_{max_token_per_img}" + '.txt'
    print(all_save_file_dir + all_save_txt_file)
    print(all_file_result)
    with open(os.path.join(all_save_file_dir, all_save_txt_file), 'w') as f:
        f.write(all_file_result)
