import csv
import dotenv
import os
from openai import OpenAI


# Framing example
# more samples
# sample diary that is not description (diary_text: ..., then answer (one for a good diary, one for a bad diary))
# review temperature, (lower, more consistant results, and want creative results)
# different model
# "stablediffusion" might be confusing, describe lighting/detail, what i want from the prompt
# few shot examples to combine different good prompts and chatgpt can remix it.
def generate_stable_diffusion_prompt(diary_text):
    s = f"""You are an AI designed to transform meal descriptions into stable diffusion prompts that create realistic images of the described food. Given a meal description, create a detailed and vivid prompt suitable for generating a realistic image.

If the description provides specific details (e.g., quantities, ingredients, or combinations), use that information precisely and accurately. Ensure that the food items are depicted exactly as described, including their amounts and any specific brands or types mentioned. Add realistic elements such as the environment (e.g., a restaurant, a kitchen, an outdoor picnic) and props (e.g., plates, utensils) to create a vivid scene.

If the description is vague or lacks detail, carefully expand on the information without contradicting the original description. Assume reasonable quantities, and add elements that enhance the realism and visual appeal, such as garnishes, complementary items (e.g., drinks or side dishes), or a fitting setting (e.g., a breakfast table or an evening picnic). However, make sure the core of the meal is depicted exactly as stated.

Always aim for a high level of realism while staying true to the original meal description.

This is an example of what a good stable diffusion prompt looks like:

"Food photography of a front-view rustic presentation of French fries in a wooden basket, the fries spilling out onto a dark linen cloth beneath. The basket is weathered and worn, adding to the rustic ambiance of the scene. The fries are arranged haphazardly, their crispy edges glistening under soft, ambient light. A sprinkle of sea salt highlights their golden hue and savory aroma. The dark backdrop sets a cozy atmosphere, enhancing the visual appeal of the fries and evoking the comforting feeling of indulging in a classic favorite." Use this as the basis of what you generate for the stable diffusion prompt.

Do not reply with anything more than the stable diffusion prompt.

The incoming meal description is the rest of this text: 
"""
    s += diary_text

    return s


def get_stablediffusion_prompt_from_openai(diary_text):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You're purpose is to take simple descriptions of a description of food in a photo and expanding it into a stablediffusion prompt.",
            },
            {"role": "user", "content": generate_stable_diffusion_prompt(diary_text)},
        ],
    )

    return str(completion.choices[0].message.content)


dotenv.load_dotenv()

MAX_COUNT = -1
counter = 0
with open(os.environ["BASE_DATA_FILE"], encoding="utf-8") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    next(tsv_file)  # Skip the header row

    for idx, row in enumerate(tsv_file):
        exploded_path = os.path.join(os.environ["OUTPUT_DIR_MULTILABEL"], f"row-{idx}")
        # Create the directory if it doesn't exist
        os.makedirs(exploded_path, exist_ok=True)

        # Write the original diary text to a file
        original_diary_text = os.path.join(exploded_path, "original_diary_text.txt")
        if not os.path.exists(original_diary_text):
            with open(original_diary_text, "w") as text_file:
                text_file.write(row[0])

        # Write the ground truth labels to a file
        ground_truth_labels = os.path.join(exploded_path, "ground_truth_labels.txt")
        if not os.path.exists(ground_truth_labels):
            with open(ground_truth_labels, "w") as text_file:
                text_file.write(row[1])

        # Use ChatGPT to generate the Stable Diffusion prompt for the diary
        stable_diffusion_prompt = os.path.join(
            exploded_path, "stable_diffusion_prompt.txt"
        )
        if not os.path.exists(stable_diffusion_prompt):
            generated_prompt = get_stablediffusion_prompt_from_openai(row[0])
            with open(stable_diffusion_prompt, "w", encoding="utf-8") as text_file:
                text_file.write(generated_prompt)

        if MAX_COUNT > 0:
            # Max count of rows to process
            counter = counter + 1
            if counter >= MAX_COUNT:
                exit(1)
