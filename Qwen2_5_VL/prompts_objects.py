###################################
###### GENERAL PROMPTS ############
###################################

mllm_objects_prompt_CoT = '''
- You are a visual-semantic reasoning expert. You are given a reference image and a modification text.
- Reference Image: The image that will be modified.
- Modification Text: Instructions that specify changes to be applied to the reference image.

## Your goal is to:
1. Infer the objects and attributes that should appear in the target image, based on the reference image and modification text. 
2. Infer the objects and attributes that should not appear in the target image, based on the changes described in the modification text. 
3. Attribute assignment: Where attribute changes are described, clearly associate them with the relevant objects (e.g., color change of a shirt).

## To complete the task accurately, please follow these steps:

    ### Describe the Reference Image ###  
    - List the objects and their attributes present in the reference image step-by-step.

    ### Understand the Modification Instructions ###  
    Analyze modification instruction step-by-step to identify changes to objects and attributes, including additions, deletions, or modifications.

    ### Apply the Modifications ### 
    - Update the objects and attributes from the reference image according to the modification instructions to obtain the expected content of the target image.
    - Please complete this task step by step.

    ### Determine the Content of the Target Image ### 
    - Existent Object (Objects and Attributes that Must Exist): 
        1. List the objects and attributes that must be present in the target image. 
        2. Be specific, especially if attributes are provided in the modification text. 
    - Nonexistent Object (Objects and Attributes that Must Not Exist): 
        1. List the objects and attributes that must not be present in the target image. 
        2. Include any objects or attributes explicitly removed or modified to no longer exist.

## On the input format <Input>
- Input consist of two parts: The reference image and the modification text.
{{
    "Reference Image": "{image_path}",
    "Modification text": "{modification_text}"
}}
    - The reference image is a path provided in the image_path field of the user content data type, which furnishes the content of the reference image.
    - The modification text is the text that describes the changes to be made to the reference image.

## Guidelines on determining the response <Response>
- The response must be composed of **two clearly separated parts**: Existent Object and Unexist Object
- The output must be formatted strictly as a valid JSON dictionary, using the following structure:
{{
    "Existent Object": [<list_of_existent_objects>],
    "Unexist Object": [<list_of_unexist_objects>]
}}

Here are some more examples for reference:

## Example 1
<Input>
{{
    "Reference Image": "/data/tangwenyue/Dataset/CIRR/dev/dev-1010-3-img1.png",
    "Modification Text": "add one more deer and add some sunlight."
}}
<Response>
{{
    "Existent Object": ["Two deer", "Brown color", "Long and curved horns", "Green grass field", "Sunlight"],
    "Unexist Object": []
}}

## Example 2
<Input>
{{
    "Reference Image": "/data/tangwenyue/Dataset/CIRR/dev/dev-274-0-img1.png",
    "Modification Text": "Shows a smaller, similarly shaped dog with lighter brown fur standing on stone tile path."
}}
<Response>
{{
    "Existent Object": ["white carriage", "daylight"],
    "Unexist Object": ["horses", "people"]
}}

## Example 3
<Input>
{{
    "Reference Image": "/data/tangwenyue/Dataset/CIRR/dev/dev-8-3-img1.png",
    "Modification Text": "Change the cart to a white, unmanned carriage in daylight with no horses."
}}
<Response>
{{
    "Existent Object": ["small dog", "lighter brown fur", "stone tile path."],
    "Unexist Object": ["white fur", "lying down", "brown patches."]
}}
'''