import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

if torch.cuda.is_available():
    device_id = 6
    torch.cuda.set_device(device_id)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/lihuan/fridge_qwen/models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map=device
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("/data/lihuan/fridge_qwen/models/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/data/tangwenyue/Code/ZS-CIR/Qwen2_5_VL/example1.jpg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]
#
# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(model.device)
#
# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/data/tangwenyue/Code/ZS-CIR/Qwen2_5_VL/demo.jpeg"},
#             {"type": "image", "image": "/data/tangwenyue/Code/ZS-CIR/Qwen2_5_VL/example1.jpg"},
#             {"type": "text", "text": "Identify the similarities between these images."},
#         ],
#     }
# ]
#
# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(device)
#
# # Inference
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)


import prompts_shared_concept
available_prompts = [f'prompts.{x}' for x in prompts_shared_concept.__dict__.keys() if '__' not in x]
prompt_sc = prompts_shared_concept.mllm_shared_concept_prompt_CoT

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000283001.jpg",
            },
            {"type": "text",
             "text": prompt_sc.format(
                 image_path="/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000283001.jpg",
                 modification_text="has a dog of a different breed and shows a jolly roger"
             )
             },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("The output text is ", output_text)

import json
shared_concept_dict = json.loads(output_text[0])
shared_concept = shared_concept_dict["Shared Concept"]
print("shared concept is", shared_concept)