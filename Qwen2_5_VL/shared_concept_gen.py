import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import copy
import os
import json
import pickle
import argparse
import numpy as np
import PIL
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from tqdm import tqdm

import datasets
import prompts_shared_concept

if torch.cuda.is_available():
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# def parser_args():
#     parser = argparse.ArgumentParser('')
#     # Base Model Choices
#     parser.add_argument("--model_name", type=str, default='qwen2_5_vl',
#                         help="Which pre-trained mllm model to use")
#     # Dataset Arguments
#     parser.add_argument("--dataset", type=str, required=True,
#                         choices=['cirr', 'circo', 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
#                                  'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 'genecis_focus_object'],
#                         help="Dataset to use")
#     parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
#                         help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
#     parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
#     # MLLM Prompt Arguments.
#     available_prompts = [f'prompts.{x}' for x in prompts_shared_concept.__dict__.keys() if '__' not in x]
#     parser.add_argument("--qwen_cir_prompt", default='prompts_shared_concept.mllm_shared_concept_prompt_CoT', type=str,
#                         choices=available_prompts,
#                         help='Denotes the base prompt to use alongside Qwen2.5-VL. Has to be available in prompts.py')
#     parser.add_argument("--batch_size", default=1, type=int, help='Batch size to use.')
#     args = parser.parse_args()
#     return args
#
# args = parser_args()


import prompts_shared_concept
available_prompts = [f'prompts.{x}' for x in prompts_shared_concept.__dict__.keys() if '__' not in x]
prompt_sc = prompts_shared_concept.mllm_shared_concept_prompt_CoT

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "<image_path>",
            },
            {"type": "text",
             "text": prompt_sc.format(
                 image_path="<image_path>",
                 modification_text="<modification_text>"
             )
             },
        ],
    }
]

def fill_messages_template(messages_template, image_path, modification_text):
    """
    替换 messages_template 中的 <image_path> 和 <modification_text> 占位符
    """
    filled = copy.deepcopy(messages_template)
    for msg in filled:
        if msg["role"] == "user":
            for block in msg["content"]:
                if block["type"] == "image":
                    block["image"] = image_path
                elif block["type"] == "text":
                    block["text"] = block["text"].replace("<image_path>", str(image_path))
                    block["text"] = block["text"].replace("<modification_text>", modification_text)
    return filled

qwen_path = "/mnt/input_zuo/pt_weights/Qwen2.5-VL-7B-Instruct/"
processor = AutoProcessor.from_pretrained(qwen_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path, torch_dtype="auto", device_map=device
)


def fiq_generate_shared_concept(dress_type, messages, processor, model, output_json):
    with open(f'/data/tangwenyue/Dataset/FashionIQ/captions/cap.{dress_type}.val.json') as f:
        triplets = json.load(f)
    print(len(triplets))

    for data in tqdm(triplets):
        reference_name = data['candidate']
        reference_image_path = f"/data/tangwenyue/Dataset/FashionIQ/images/{reference_name}.jpg"
        rel_caps = data['captions']
        rel_caps = np.array(rel_caps).T.flatten().tolist()
        relative_captions = " and ".join(f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps) - 1, 2))

        composed_messages = fill_messages_template(messages, reference_image_path, relative_captions)
        composed_text = processor.apply_chat_template(
            composed_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(composed_messages)
        inputs = processor(
            text=[composed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # shared_concept_dict = json.loads(output_text[0])
        # shared_concept = shared_concept_dict["Shared Concept"]

        try:
            shared_concept_dict = json.loads(output_text[0])
            shared_concept = shared_concept_dict["Shared Concept"]
        except Exception as e:
            shared_concept = "<PARSE_ERROR>"
            print(f"Warning: Failed to parse output: {output_text[0]}")
        data['shared_concept'] = shared_concept
        # print("The shared concept is", shared_concept)
        torch.cuda.empty_cache()

    with open(output_json, 'w') as f:
        json.dump(triplets, f, indent=4)


CIRR_root = "/mnt/input_zuo/ZS-CIR/CIRR"
def cirr_generate_shared_concept(messages, processor, model, output_json):
    with open(f'{CIRR_root}/cirr/captions/cap.rc2.val.json', 'r') as f:
        triplets = json.load(f)
    print(len(triplets))

    with open(f'{CIRR_root}/cirr/image_splits/split.rc2.val.json') as f:
        name_to_relpath = json.load(f)

    for data in tqdm(triplets):
        relative_caption = data['caption']
        reference_name = data['reference']
        reference_image_path = os.path.join(CIRR_root, name_to_relpath[reference_name])

        composed_messages = fill_messages_template(messages, reference_image_path, relative_caption)
        composed_text = processor.apply_chat_template(
            composed_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(composed_messages)
        inputs = processor(
            text=[composed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # shared_concept_dict = json.loads(output_text[0])
        # shared_concept = shared_concept_dict["Shared Concept"]

        try:
            shared_concept_dict = json.loads(output_text[0])
            shared_concept = shared_concept_dict["Shared Concept"]
        except Exception as e:
            shared_concept = "<PARSE_ERROR>"
            print(f"Warning: Failed to parse output: {output_text[0]}")
        data['shared_concept'] = shared_concept
        # print("The shared concept is", shared_concept)
        torch.cuda.empty_cache()

    with open(output_json, 'w') as f:
        json.dump(triplets, f, indent=4)


def circo_generate_shared_concept(messages, processor, model, output_json):
    with open('/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', 'r') as f:
        imgs_info = json.load(f)

    img_paths = [os.path.join('/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017', img_info["file_name"]) for img_info in imgs_info["images"]]
    img_ids = [img_info["id"] for img_info in imgs_info["images"]]
    img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}

    with open('/data/tangwenyue/Dataset/CIRCO/annotations/val.json') as f:
        annotations = json.load(f)

    for data in tqdm(annotations):
        query_id = str(data['id'])
        
        relative_caption = data['relative_caption']
        reference_img_id = data['reference_img_id']
        # self.img_paths[self.img_ids_indexes_map[reference_img_id]]
        reference_image_path = img_paths[img_ids_indexes_map[reference_img_id]]

        composed_messages = fill_messages_template(messages, reference_image_path, relative_caption)
        composed_text = processor.apply_chat_template(
            composed_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(composed_messages)
        inputs = processor(
            text=[composed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # shared_concept_dict = json.loads(output_text[0])
        # shared_concept = shared_concept_dict["Shared Concept"]

        try:
            shared_concept_dict = json.loads(output_text[0])
            shared_concept = shared_concept_dict["Shared Concept"]
        except Exception as e:
            shared_concept = "<PARSE_ERROR>"
            print(f"Warning: Failed to parse output: {output_text[0]}")
        data['new_shared_concept'] = shared_concept
        # print("The shared concept is", shared_concept)
        torch.cuda.empty_cache()

    with open(output_json, 'w') as f:
        json.dump(triplets, f, indent=4)


# Process the data_process
# datasets = ['fashioniq dress', 'fashioniq shirt', 'fashioniq toptee', 'cirr']
datasets = ['cirr']
for data in datasets:
    if 'fashioniq' in data:
        data, fiq_data_type = data.split(' ')
        assert fiq_data_type in ['dress', 'shirt', 'toptee']

        fiq_output_json = f"/data/tangwenyue/Code/ZS-CIR/Qwen2_5_VL/cap.{fiq_data_type}.val.json"
        fiq_generate_shared_concept(fiq_data_type, messages, processor, model, fiq_output_json)
        print(f"Successfully process {data}_{fiq_data_type}!")
    elif data == 'cirr':
        cirr_output_json = f"{CIRR_root}/cap.rc2.val.json"
        cirr_generate_shared_concept(messages, processor, model, cirr_output_json)
        print(f"Successfully process {data}!")
    elif data == 'circo':
        circo_output_json = "/data/tangwenyue/Code/ZS-CIR/Qwen2_5_VL/val.json"
        circo_generate_shared_concept(messages, processor, model, circo_output_json)
        print(f"Successfully process {data}!")
