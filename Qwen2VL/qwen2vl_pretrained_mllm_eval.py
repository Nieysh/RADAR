import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import random
import numpy as np
import glob
from tqdm import tqdm
import json
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

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default='.')
args = parser.parse_args()

##########  model #############
# default: Load the model on the available device(s)
model_path = args.model_path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,  device_map="auto" # torch_dtype="auto",
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 768*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

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
    save_file = model_path.split('/')[-1] + f"_{min_pixels}~{max_pixels}" + '.json'
    print(save_file_dir + '/' + save_file)
    if os.path.exists(save_file_dir + '/' + save_file):
        print(f"{save_file_dir + '/' + save_file} exists, skip!")
        continue

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
            question = replace_image_with_image_token(data[q_id]["question"], all_images, '<|vision_start|><|image_pad|><|vision_end|>')
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
                        f"{chr(ord('A') + i)}. {replace_image_with_image_token(option, extract_images_from_text(option), '<|vision_start|><|image_pad|><|vision_end|>')}"
                        for i, option in enumerate(options)]
                else:
                    options = distraction + [data[q_id]["answer"]]
                    for option in options:
                        img_files.extend(
                            get_image_files_from_all_images(pretrain_benchmark_img_dir + f"/{img_cate}",
                                                            extract_images_from_text(option), q_id))
                    options_letters = ["B", "A"]
                    options = [
                        f"{chr(ord('A') + i)}. {replace_image_with_image_token(option, extract_images_from_text(option), '<|vision_start|><|image_pad|><|vision_end|>')}"
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

        messages = [
            {
                "type": "image",
                "content": [
                    {
                        "type": "image",
                        "image": img_file,
                    },
                ],
            } for img_file in img_files
        ]
        messages.append({"type": "text", "text": question, "content": []})

        # add img tokens
        if "MMMU_Pro" in pretrain_benchmark_file:
            text = question
        else:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_len_wo_answer = inputs['input_ids'].size(1)

        texts = []
        for answer in answers:
            query = text + " " + answer
            texts.append(query)
        image_inputs, video_inputs = process_vision_info([messages for _ in answers])

        processor.tokenizer.padding_side = 'left'
        batch_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch_inputs = batch_inputs.to("cuda")
        batch_padding_num = (batch_inputs['input_ids'][:,:] == 151643).sum(dim=1).cpu().numpy()
        batch_input_len_wo_answer = [input_len_wo_answer + pad_num for pad_num in batch_padding_num]

        with torch.inference_mode():
            outputs = model(**batch_inputs, return_dict=True)

        logits = outputs['logits']
        answers_avg_logits = []
        answers_ori_logits = []
        for i, answer in enumerate(answers):
            answer_len_real = 0
            answer_logits = []
            for idx in range(batch_input_len_wo_answer[i] - 1, logits[i].size(0) - 1):
                tok_logit = logits[i, idx, batch_inputs['input_ids'][i, batch_input_len_wo_answer[i] + answer_len_real]].item()
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
    file_result = f"Success Rate: {total_success}/{len(q_ids)}:{accuracy * 100:.2f}%, Answer Prob: {avg_prob}\n"
    all_cat_res[category.split('_pretrained_mllm_eval')[0]] = {
        "accuracy": accuracy,
        "avg_prob": avg_prob
    }


    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir, exist_ok=True)
    with open(os.path.join(save_file_dir, save_file), 'w') as f:
        json.dump(all_res, f)

    save_txt_file = model_path.split('/')[-1] + f"_{min_pixels}~{max_pixels}" + '.txt'
    with open(os.path.join(save_file_dir, save_txt_file), 'w') as f:
        f.write(file_result)
    print(save_file_dir + '/' + save_txt_file)
    print(file_result)

if "MMBench" in pretrain_benchmark_file or "MMMU_Pro" in pretrain_benchmark_file:
    all_save_file_dir = os.path.join(save_dir, 'total')
    all_save_file = model_path.split('/')[-1] + f"_{min_pixels}~{max_pixels}" + '.json'
    print(all_save_file_dir + '/' + all_save_file)
    if not os.path.exists(all_save_file_dir):
        os.makedirs(all_save_file_dir, exist_ok=True)
    with open(os.path.join(all_save_file_dir, all_save_file), 'w') as f:
        json.dump(all_cat_res, f)

    all_avg_acc = sum(all_cat_res[k]['accuracy'] for k in all_cat_res.keys()) / len(all_cat_res)
    all_avg_prob = sum(all_cat_res[k]['avg_prob'] for k in all_cat_res.keys()) / len(all_cat_res)
    all_file_result = f"Average Success Rate: {sum(all_cat_res[k]['accuracy'] for k in all_cat_res.keys())}/{len(all_cat_res)}:{all_avg_acc*100:.2f}%, Average Answer Prob: {all_avg_prob}\n"
    all_save_txt_file = model_path.split('/')[-1] + f"_{min_pixels}~{max_pixels}" + '.txt'
    print(all_save_file_dir + all_save_txt_file)
    print(all_file_result)
    with open(os.path.join(all_save_file_dir, all_save_txt_file), 'w') as f:
        f.write(all_file_result)