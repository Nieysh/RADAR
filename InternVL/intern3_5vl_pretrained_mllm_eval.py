import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import sys
try:
    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
    from internvl.conversation import get_conv_template
except:
    sys.path.append("YOUR/PATH/TO/InternVL/internvl_chat")

from transformers import AutoModel, AutoTokenizer
import glob
import json
import random
from tqdm import tqdm
import ast
import re
import argparse

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(0)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

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

def load_txt(file):
    with open(file, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
    return data

def extract_from_txt(data, mode):
    if mode == 'acc':
        x = data.find('Success Rate: ') + len('Success Rate: ')
        y = data.find(':', x)
        metric = float(eval(data[x:y]))
    if mode == 'answer_prob':
        x = data.find('Answer Prob: ') + len('Answer Prob: ')
        metric = float(data[x:])
    return metric

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default='.')
args = parser.parse_args()


##########  model #############
model_path = args.model_path
print(model_path)
config = InternVLChatConfig.from_pretrained(model_path)
model = InternVLChatModel.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
model.img_context_token_id = img_context_token_id
device = torch.device(model.language_model.device if torch.cuda.is_available() else 'cpu')
max_num = 3
max_token_per_img = max_num * 256
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
    save_file = model_path.split('/')[-1] + f"_{max_token_per_img}" + '.json'
    save_txt_file = model_path.split('/')[-1] + f"_{max_token_per_img}" + '.txt'

    print(save_file_dir + '/' + save_file)
    # if os.path.exists(save_file_dir + '/' + save_file):
    #     print(f"{save_file_dir + '/' + save_file} exists, skip!")
    #     continue

    for q_id in tqdm(q_ids):
        torch.cuda.empty_cache()
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
            question = replace_image_with_image_token(data[q_id]["question"], all_images,
                                                      '<image>')
            if extract_images_from_text(data[q_id][
                                            "ori_options"]):  # if images appear in options, then include options in question prompt
                distraction = item['distraction']
                if random.random() > 0.5:
                    options = [data[q_id]["answer"]] + distraction
                    for option in options:
                        img_files.extend(
                            get_image_files_from_all_images(pretrain_benchmark_img_dir + f"/{img_cate}",
                                                            extract_images_from_text(option), q_id))
                    options_letters = ["A", "B"]
                    options = [
                        f"{chr(ord('A') + i)}. {replace_image_with_image_token(option, extract_images_from_text(option), '<image>')}"
                        for i, option in enumerate(options)]
                else:
                    options = distraction + [data[q_id]["answer"]]
                    for option in options:
                        img_files.extend(
                            get_image_files_from_all_images(pretrain_benchmark_img_dir + f"/{img_cate}",
                                                            extract_images_from_text(option), q_id))
                    options_letters = ["B", "A"]
                    options = [
                        f"{chr(ord('A') + i)}. {replace_image_with_image_token(option, extract_images_from_text(option), '<image>')}"
                        for i, option in enumerate(options)]
                question = question + " " + ' '.join(options) + " Please select the correct option: "
        else:
            question = item['question']

        # answer
        if 'MMMU_Pro' in pretrain_benchmark_file:
            if extract_images_from_text(data[q_id]["ori_options"]):
                answers = options_letters
            else:
                distraction = item['distraction']
                answers = [item["answer"]] + distraction
        else:
            answer = item['answer']
            distraction = item['distraction']
            answers = [answer] + distraction

        print(f"Q_ID: {q_id} Question: {question} Candidates: {answers}")
        if len(img_files) > 12:
            print(f"Too many images for {q_id}, skip!")
            continue

        pixel_values_list = []
        for img_file in img_files:
            pixel_values_list.append(load_image(img_file, max_num=max_num).to(torch.bfloat16))
        num_patches_list = [pixel_values1.size(0) for pixel_values1 in pixel_values_list]
        pixel_values = torch.cat(pixel_values_list, dim=0)

        if pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        if "MMBench" in pretrain_benchmark_file or "MMMU_Pro" in pretrain_benchmark_file or "MathVista" in pretrain_benchmark_file:
            sep = '\n'
        else:
            sep = ' '
        for idx, num_patches in enumerate(num_patches_list):
            if pixel_values is not None and '<image>' not in question:
                question = f'<image>{sep}' + question
            image_tokens = '<img>' + '<IMG_CONTEXT>' * model.num_image_token * num_patches + '</img>'
            question = question.replace('<image>', image_tokens, 1)

        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs_wo_answer = tokenizer(query, return_tensors='pt')
        input_len_wo_answer = model_inputs_wo_answer['input_ids'].size(1)

        queries = []
        batch_pixel_values_list = []
        for answer in answers:
            queries.append(query + str(answer))
            batch_pixel_values_list.extend(pixel_values_list)
        batch_num_patches_list = [pixel_values2.size(0) for pixel_values2 in batch_pixel_values_list]
        batch_pixel_values = torch.cat(batch_pixel_values_list, dim=0).to(device)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(device)
        batch_padding_num = (input_ids[:,:] == 151643).sum(dim=1).cpu().numpy()
        batch_input_len_wo_answer = [input_len_wo_answer + pad_num for pad_num in batch_padding_num]
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        with torch.inference_mode():
            output = model(
                pixel_values=batch_pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor([1] * sum(batch_num_patches_list), dtype=torch.long),
                return_dict=True
            )

        logits = output['logits']
        answers_avg_logits = []
        answers_ori_logits = []
        for i, answer in enumerate(answers):
            answer_len_real = 0
            answer_logits = []
            for idx in range(batch_input_len_wo_answer[i] - 1, logits[i].size(0) - 1):
                tok_logit = logits[i, idx, input_ids[i, batch_input_len_wo_answer[i] + answer_len_real]].item()
                answer_logits.append(tok_logit)
                answer_len_real += 1
            answers_avg_logits.append(sum(answer_logits) / len(answer_logits))
            answers_ori_logits.append(answer_logits.copy())
        answer_prob = torch.softmax(torch.tensor(answers_avg_logits), dim=0)
        success = answer_prob.argmax().item() == 0
        if success:
            total_success += 1

        res = {
            "q_id": q_id,
            "question": item['question'],
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

    with open(os.path.join(save_file_dir, save_txt_file), 'w') as f:
        f.write(file_result)
    print(save_file_dir + '/' + save_txt_file)
    print(file_result)


if "MMBench" in pretrain_benchmark_file or "MMMU_Pro" in pretrain_benchmark_file:
    all_save_file_dir = os.path.join(save_dir, 'total')
    all_save_file = model_path.split('/')[-1] + f"_{max_token_per_img}" + '.json'
    print(all_save_file_dir + '/' + all_save_file)
    if not os.path.exists(all_save_file_dir):
        os.makedirs(all_save_file_dir, exist_ok=True)
    with open(os.path.join(all_save_file_dir, all_save_file), 'w') as f:
        json.dump(all_cat_res, f)

    all_avg_acc = sum(all_cat_res[k]['accuracy'] for k in all_cat_res.keys()) / len(all_cat_res)
    all_avg_prob = sum(all_cat_res[k]['avg_prob'] for k in all_cat_res.keys()) / len(all_cat_res)
    all_file_result = f"Average Success Rate: {sum(all_cat_res[k]['accuracy'] for k in all_cat_res.keys())}/{len(all_cat_res)}:{all_avg_acc*100:.2f}%, Average Answer Prob: {all_avg_prob}\n"
    all_save_txt_file = model_path.split('/')[-1] + f"_{max_token_per_img}" + '.txt'
    print(all_save_file_dir + all_save_txt_file)
    print(all_file_result)
    with open(os.path.join(all_save_file_dir, all_save_txt_file), 'w') as f:
        f.write(all_file_result)
