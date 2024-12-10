import dotenv
import os
import json

dotenv.load_dotenv()

base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]

image_to_label = {}

with os.scandir(base_directory) as entries:
    for entry in entries:
        if entry.is_dir():
            # load up the updated ground truth labels
            subdir = os.path.join(base_directory, entry.name)
            updated_ground_truth_labels_file = os.path.join(
                subdir, "ground_truth_labels_updated.txt"
            )
            if not os.path.exists(updated_ground_truth_labels_file):
                print(f"Skipping {subdir} as ground truth labels do not exist")
                continue
            with open(updated_ground_truth_labels_file, "r") as f:
                updated_ground_truth_labels = f.read()

            # load up the image file
            image_directory = os.path.join(subdir, "output")
            # Walk through the image directory
            for root, _, files in os.walk(image_directory):
                # Prep for multiple images, possible, in the future
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_found = os.path.join(root, file)

                        # add the image to the dictionary
                        # map images to their updated ground truth labels
                        # multiple images can be generated for the same diary and have the same labels
                        image_to_label[image_found] = updated_ground_truth_labels

# save the json file
output_file = os.path.join(base_directory, "image_to_label.json")
print(f"Saving image to label mapping to {output_file}")
with open(output_file, "w") as f:
    f.write(json.dumps(image_to_label))
