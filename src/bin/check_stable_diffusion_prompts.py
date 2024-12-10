# iterate over the multi-label directory
# for each file, read the stable_diffusion_prompt.txt file, check the length of the prompt

import os
import dotenv

dotenv.load_dotenv()

# Looked at 80 generated prompts, found the smallest characters (508),
# reduce it further and anything below this needs to be investigated
MINIMUM_PROMPT_CHARS_COUNT = 400


# used this to find `MINIMUM_PROMPT_CHARS_COUNT`
def find_smallest_character_count(root_dir):
    smallest_count = float("inf")
    smallest_file = ""

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "stable_diffusion_prompt.txt" in filenames:
            file_path = os.path.join(dirpath, "stable_diffusion_prompt.txt")
            try:
                with open(file_path, "r") as file:
                    content = file.read()
                    char_count = len(content)
                    if char_count < smallest_count:
                        smallest_count = char_count
                        smallest_file = file_path
            except IOError as e:
                print(f"Error reading file {file_path}: {e}")

    if smallest_count != float("inf"):
        print(f"The smallest character count is {smallest_count}")
        print(f"Found in file: {smallest_file}")


def check_prompts_length_for_problems(root_dir, min_chars_count):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "stable_diffusion_prompt.txt" in filenames:
            file_path = os.path.join(dirpath, "stable_diffusion_prompt.txt")
            try:
                with open(file_path, "r") as file:
                    content = file.read()
                    char_count = len(content)
                    if char_count < min_chars_count:
                        print(
                            f"Found a prompt with {char_count} characters in file {file_path}"
                        )
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")


# find_smallest_character_count(os.environ["OUTPUT_DIR_MULTILABEL"])
check_prompts_length_for_problems(
    os.environ["OUTPUT_DIR_MULTILABEL"], MINIMUM_PROMPT_CHARS_COUNT
)
