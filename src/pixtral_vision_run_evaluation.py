from categories import get_categories
from dotenv import load_dotenv
import json
import os
import base64
import csv
from config import Config
from multi_label_stratified import multi_label_stratified_split
import time
from vllm import LLM
from vllm.sampling_params import SamplingParams

load_dotenv()


def invoke_model(prompt, image_data_b64):

    model_name = "mistralai/Pixtral-12B-2409"
    sampling_params = SamplingParams(max_tokens=8192)
    llm = LLM(model=model_name, tokenizer_mode="mistral")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"image_url": f"data:image/jpeg;base64,{image_data_b64}"},
            ],
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)


def run_inference(image, usda_categories):
    with open(image, "rb") as f:
        image_data = f.read()
        image_data_b64 = base64.b64encode(image_data).decode("utf-8")

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

    response_message = invoke_model(prompt, image_data_b64)
    return response_message


if __name__ == "__main__":
    categories = get_categories()
    config = Config()

    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    json_file = os.path.join(base_directory, "image_to_label.json")
    image_to_label = json.load(open(json_file))

    results = []  # store the results from openai to write to a file
    result_file = os.path.join(base_directory, "pixtral_results.csv")

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
    for image_path, labels in test_data:
        if count % 10 == 0:
            print(f"Count: {count} of {MAX_COUNT}")
        response_from_openai = run_inference(image_path, categories)
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
