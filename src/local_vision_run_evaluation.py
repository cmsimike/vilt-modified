from categories import get_categories
from dotenv import load_dotenv
import json
import os
import base64
import csv
from config import Config
from multi_label_stratified import multi_label_stratified_split
import time
import io

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

load_dotenv()


def get_qwen():
    name = "qwen2-7b"
    model_id = "Qwen/Qwen2-VL-72B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return name, model, processor


def invoke_qwen(model, processor, prompt_long, image_data):
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_b64}"},
                {"type": "text", "text": "Describe this image."},
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
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)
    return output_text


def get_molmo():
    name = "Molmo-7b"
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    return name, model, processor


def invoke_molmo(model, processor, prompt_long, image_data):

    image = Image.open(io.BytesIO(image_data))

    inputs = processor.process(images=[image], text=prompt_long)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )
    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    # print the generated text
    print(generated_text)

    return generated_text


def get_llama():
    name = "llama-11b"
    model_id = "meta-llama/Llama-3.2-11B-Vision"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return name, model, processor


def invoke_llama(model, processor, prompt_long, image_data):

    image = Image.open(io.BytesIO(image_data))

    prompt = "<|image|><|begin_of_text|>" + prompt_long
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=50)
    model_response = processor.decode(output[0])
    return model_response


def run_inference(model, processor, image, usda_categories):
    with open(image, "rb") as f:
        image_data = f.read()

    # Prepare the prompt
    usda_categories_str = ", ".join(usda_categories)
    prompt = f"""
    You will be shown an image. Please identify the food items in the image that belong to the following food categories between equal signs
    =========
    {usda_categories_str}
    =========
    Do not mention any additional categories or food items not in the list.
    Only respond with a comma separated list of the categories you see in the image from the list provided. Your response should be only the comma separated list with no additional text as part of the response.
    """

    response_message = invoke_qwen(model, processor, prompt, image_data)
    return response_message


if __name__ == "__main__":
    categories = get_categories()
    config = Config()

    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    json_file = os.path.join(base_directory, "image_to_label.json")
    image_to_label = json.load(open(json_file))

    data = []
    for image_path, label in image_to_label.items():
        tuple_data = (image_path, [x for x in label.split(",")])
        data.append(tuple_data)

    # Split the data
    _, test_data, _ = multi_label_stratified_split(
        data,
        test_size=(1 - config.TRAIN_SPLIT),
        random_state=config.RANDOM_SEED,
    )

    count = 0
    MAX_COUNT = 100
    start_time = time.time()

    name, model, processor = get_qwen()
    results = []  # store the results from openai to write to a file
    result_file = os.path.join(base_directory, f"{name}_results.csv")

    for image_path, labels in test_data:
        if count % 10 == 0:
            print(f"Count: {count} of {MAX_COUNT}")
        response_from_openai = run_inference(model, processor, image_path, categories)
        results.append(
            {
                "image_path": image_path,
                "actual_labels": labels,
                "predicted_labels": response_from_openai,
            }
        )

        count = count + 1
        if count >= MAX_COUNT:
            break

    end_time = time.time()
    execution_time = end_time - start_time  # Calculate the duration in seconds
    print(f"{MAX_COUNT} iterations took {execution_time:.4f} seconds to complete.")

    with open(result_file, "w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=",")
        dw.writeheader()
        dw.writerows(results)

    print(f"Results written to {result_file}")
