from categories import get_categories
from dotenv import load_dotenv
import json
import os
import base64
import csv
from pydantic import BaseModel
from openai import OpenAI
from config import Config
from multi_label_stratified import multi_label_stratified_split
import time

load_dotenv()


class Category(BaseModel):
    category: str
    ingredients: list[str]


class FoundCategories(BaseModel):
    categories: list[Category]


def invoke_openai(image, usda_categories):
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
    Only respond with a JSON array of the categories you see in the image from the list provided.
    Do not mention any additional categories or food items not in the list.
    Your response should be only the JSON array with no additional text as part of the response.
    Each category should be from the list of food categories given and returned in the property named `category`. If you can identity the ingredients for a category (the ingredients should also be from the categories given only), list those in the property `ingredients`. For instance {{'category': 'sandwich', 'ingredients': ['bread', 'cheese', 'tomato']}}. If not additional ingredients are identified, you can leave the `ingredients` property as an empty list.

    Both ingredients of categories and categories themselves should come from the list of previously given categories between equal signs.
    """
    # print("Prompt:", prompt)
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        # TODO We should test both this model and the GPT-4o-mini model
        # https://platform.openai.com/docs/guides/structured-outputs/introduction
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{'image/jpeg'};base64,{image_data_b64}"
                        },
                    },
                ],
            }
        ],
        response_format=FoundCategories,
    )

    response_message = response.choices[0].message.content

    try:
        categories_found = json.loads(response_message)["categories"]
        return categories_found
    except json.JSONDecodeError:
        print(f"actual response: {response_message}")
        print("Failed to parse the assistant's reply as JSON.")


if __name__ == "__main__":
    # make sure I don't accidentally run this script
    raise Exception("Are you sure you want to re-run this script?")
    # Moved file opening up here to sanity check I've closed the file so I don't burn through OpenAI credits
    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    json_file = os.path.join(base_directory, "image_to_label.json")
    image_to_label = json.load(open(json_file))

    results = []  # store the results from openai to write to a file
    result_file = os.path.join(base_directory, "openai_results_extra_structure.csv")
    with open(result_file, "w", newline="", encoding="utf-8") as f:
        categories = get_categories()
        config = Config()

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
            response_from_openai = invoke_openai(image_path, categories)
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

        dw = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=",")
        dw.writeheader()
        dw.writerows(results)

        print(f"Results written to {result_file}")
