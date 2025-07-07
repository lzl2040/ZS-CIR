###################################
###### SHARED CONCEPTS PROMPTS ############
###################################

# mllm_shared_concept_prompt_CoT = '''
# - You are an visual-semantic reasoning expert. You are given an reference image and modification text.
# - Reference Image: The image that will be modified.
# - Modification Text: Instructions that specify changes to be applied to the reference image.
#
# - Your goal is to:
# 1. Analyze the semantic relationship between the reference image and the modification text.
# 2. Perform structured reasoning to determine what elements constitute the shared concept between the original and modified image.
#
# ## Guidelines on generating the Shared Concept Description
#
#     - Interpret the reference image to identify its key visual elements, e.g., objects, attributes, spatial relationships, and overall scene context.
#     - Analyze the modification text to understand how these elements are intended to be transformed, such as addition, negation, spatial relations, or viewpoint.
#     - Explicitly distinguish between preserved, transformed, and removed components.
#
#     - Based on the combined understanding of the reference image and the modification instruction, infer what objects and attributes are expected to remain present in the target image.
#     - From the inferred composition of the target image, extract only the semantic overlap——the part that is shared with the reference image regardless of modification.
#
#     - The shared concept is not a full description of either image, but a precise representation of what remains consistent across both.
#     - Each time generate one shared concept description only. Keep the shared concept description as short as possible. Here are some more examples for reference:
#
# ## On the input format <Input>
# - Input consist of two parts: The reference image and the modification text.
# {{
#     "Reference Image": "{image_path}"
#     "Modification Text": "{modification_text}"
# }}
#     - The reference image is a path provided in the image_path field of the user content data type, which furnishes the content of the reference image.
#     - The modification text is the text that describes the changes to be made to the reference image.
#
# ## Guidelines on determining the response <Response>
# - Responses should include only the shared_concept field, which captures the core semantic content shared between the reference image and its modified version.
# {{
#     "Shared Concept": <shared_concept>
# }}
#
# Here are some more examples for reference:
#
# ## Example 1
# <Input>
# {{
#     "Reference Image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000272126.jpg",
#     "Modification Text": "is white with a blue floral pattern"
# }}
# <Response>
# {{
#     "Shared concept": "a close-up vase with flowers in it"
# }}
#
# ## Example 2
# <Input>
# {{
#     "Reference Image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000065291.jpg",
#     "Modification Text": "has the same shape, has Arabic writing on it and the photo is taken at night"
# }}
# <Response>
# {{
#     "Shared concept": "an octagonal stop sign"
# }}
#
# ## Example 3
# <Input>
# {{
#     "Reference Image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000277771.jpg",
#     "Modification Text": "has a different color and there is a person touching it with their hands"
# }}
# <Response>
# {{
#     "Shared concept": "a fire hydrant in the foreground"
# }}
#
# '''


mllm_shared_concept_prompt_CoT = '''
- You are a visual-semantic reasoning expert. You are given a reference image and a modification text.

## Definitions

    - Reference Image: The original image that will be modified.
    - Modification Text: A natural language instruction describing specific changes that should be applied to the reference image. These changes may include the addition, removal, transformation, or reorientation of visual elements.

## Your goal is to:
1. Analyze the **semantic relationship** between the reference image and the modification text.
2. Generate a concise description of the **shared concept** — that is, the semantic content that remains **unchanged** in both the reference image and the modified image.

## To do this, follow this 3-step reasoning process:

    ### Step1: Understand the Reference Image
    - Identify the main objects and their attributes (such as color, size, position), and the overall scene structure.
    - Think about what the image is *mainly* showing.
    
    ### Step2: Interpret the Modification Text
    - Determine what specific changes are described: What is added? What is removed? What is modified (viewpoint, context, lighting, interaction, etc.)?
    - Pay attention to what is **explicitly negated or changed**.
    
    ### Step3: Derive the Shared Concept
    - Based on the above reasoning, identify the **semantic core** that persists in both versions.
    - Do **not** repeat changes from the modification text.
    - Do **not** re-describe the entire image.
    - Only extract the invariant and meaningful visual concept.

## On the input format <Input>
- Input consist of two parts: The reference image and the modification text.
{{
    "Reference Image": "{image_path}"
    "Modification Text": "{modification_text}"
}}
    - The reference image is a path provided in the image_path field of the user content data type, which furnishes the content of the reference image.
    - The modification text is the text that describes the changes to be made to the reference image.

## Guidelines on determining the response <Response>
- Responses should include only the shared_concept field, which captures the core semantic content shared between the reference image and its modified version.
{{
    "Shared Concept": <shared_concept>
}}

Here are some more examples for reference:

## Example 1
<Input>
{{
    "Reference Image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000272126.jpg",
    "Modification Text": "is white with a blue floral pattern"
}}
<Response>
{{
    "Shared concept": "a close-up vase with flowers in it"
}}

## Example 2
<Input>
{{
    "Reference Image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000065291.jpg",
    "Modification Text": "has the same shape, has Arabic writing on it and the photo is taken at night"
}}
<Response>
{{
    "Shared concept": "an octagonal stop sign"
}}

## Example 3
<Input>
{{   
    "Reference Image": "/data/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017/000000277771.jpg",
    "Modification Text": "has a different color and there is a person touching it with their hands"
}}
<Response>
{{
    "Shared concept": "a fire hydrant in the foreground"
}}

'''