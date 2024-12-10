# Project

This repository is an experiment in the ability to fine-tune a Vision-and-Language Transformer (ViLT) model to predict food categories from an image. This repository contains utilities (that can be run individually) that prepare a data set for model fine tuning and evaluation as well as the definition for the model and train/evaluation loops.

The overall architecture used a [previously trained Vilt](https://huggingface.co/dandelin/vilt-b32-mlm) as a base for our Pytorch neural network and adds a new classification layer as the final layer the same size as our total categories (after post-processing). We fine-tune the neural network by:
- Processing the image using `ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")`
- Presenting the image to the neural network with the prompt-template/question "What foods are present in this image?"
- Using multi-label output to predict what categories are in the image

# Quickstart
1. Download the data [here (need to request access)](https://lmu.app.box.com/folder/292327189640) and extract to any location.
2. Copy `env.template` to `.env` and update the values:
    1. `BASE_DATA_FILE` is the direct location of the `main_dataset_multilabel.tsv` file from the ZIP. Only really needed if you need to reprocess the "exploded directories" which the ZIP includes.
    2. `OUTPUT_DIR_MULTILABEL` points directly to the `multi-label` directory inside of the expanded ZIP file.
    3. `OPENAI_API_KEY` is the OpenAI key, only required if you're using the OpenAI scripts.
    4. `MLFLOW_TRACKING_SERVER`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are only required if using a remote MLFlow server. Can be left empty. More information on one way to setup [MLFlow](https://github.com/sachua/mlflow-docker-compose).
3. `pip install -r requirements.txt` for a majority of the functionality, `pip install -r requirements_pixtral.txt` for the pixtral requirements (not tested, doesn't work on Windows), and `pip install -r requirements_small_lms.txt` for the local language model tests. `requirements.txt` is tested against Windows and likely needs some changes for Linux.

# Process
We convert the base data file into what we've called "the exploded format" - turning a TSV file into subdirectories (named `row-{idx}`) that represent the data from the TSV. Each directory contains:
- `original_diary_text.txt` - The original diary entry from a person.
- `ground_truth_labels.txt` - The original categories from the USDA
- `stable_diffusion_prompt.txt` - A StableDiffusion prompt generated from querying ChatGPT.
- `ground_truth_labels_updated.txt` - An updated ground-truth list, converting some categories into a more generic category.
- `output/` - A directory that contains images (AI generated or otherwise) that represent the ground truth categories.

The data directory is then processed into a json file where the key is an image path and value is an array of USDA categories. This json is then used for training and validation in `train.py`.

# The Base File

The base file used is named `main_dataset_multilabel.tsv` and contains 4 columns:
- `diaries` - The raw textual representation of a meal.
- `usda_id` - The base USDA categories for each meal description. These are the categories before many categories were grouped together into more generic categories
- `Image_id` - Unused
- `category_id` - Unused

# The Json File

The json file is setup with an image path as the key to a list of categories as the value.
```json
{
    "/path/to/images/multi-label/row-0/output/a1386447-1242-4104-97b1-1adbe1e738be.png": "bread,bologna,pickles", 
    "/path/to/images/multi-label/row-1/output/3dc8ed95-8f83-4fbe-adfc-ee5ef845fbc1.png": "bread,salami,cheese,pickles,tomatoes", 
}
```
This is directly processed and used by the fine tuning process.

# Main files
Located in the `src` directory, these files are the meat of this project, with various experiments that can be run, from testing and evaluating models to experimenting with OpenAI's vision, to experimenting with local LLMs.

- `categories.py` - This Python file maps specific dataset categories to more general ones and provides utilities to retrieve unique categories and total category count. This data is constant after discussions.
- `config.py` - This Python class defines a Config object to manage hyperparameters and settings for fine-tuning a ViLT model.
- `dataset.py` - This code implements a FoodDataset class for multi-label food image classification and a create_dataloaders function to preprocess data, perform stratified splitting, and create PyTorch DataLoaders for training and testing.
- `labels.py` -  This code defines a LabelProcessor class to handle conversions between text labels and one-hot tensors for multi-label classification, leveraging the dataset's predefined categories.
- `local_vision_run_evaluation.py` - This script evaluates multi-label food image classification models by running inference on a stratified test dataset. It processes images, invokes selected models (Qwen, Molmo, or Llama) to predict categories, and writes the results to a CSV file for analysis.
- `local_vision_score_csv.py` - This script evaluates multi-label classification models by comparing predicted labels to ground truth using precision, recall, and F1 scores. It processes CSV results from multiple models and calculates performance metrics for overall and per-category evaluation.
- `model.py` - Defines a PyTorch model using ViLT for multi-label classification with an optional weight-freezing.
- `multi_label_stratified.py` - Performs multi-label stratified splitting of data into train and test sets, ensuring label distributions are preserved.
- `openai_embedding_score_results.py` - Compares transformer models for label prediction using embedding similarity, supporting various pooling strategies and logging results with MLflow.
- `openai_score_csv_results_more_structure.py` - Analyzes multi-label classification predictions by computing precision, recall, and F1-scores for overall and per-label performance, while filtering predictions to match valid categories.
- `openai_score_csv_results.py` - Analyzes multi-label classification predictions by computing precision, recall, and F1-scores for overall and per-label performance, while filtering predictions to match valid categories.
- `openai_vision_run_evaluation_more_structure.py` - Uses OpenAI's API to identify food categories and ingredients in images, structured as JSON, for multi-label classification analysis.
- `openai_vision_run_evaluation.py` - Uses OpenAI's API to identify food categories and ingredients in images, structured as JSON, for multi-label classification analysis.
- `pixtral_vision_run_evaluation.py` - Never run, but should use Pixtral to evaluate images.
- `train.py` - Fine-tunes the ViLT model for multi-label classification, logs metrics and predictions with MLflow with hyperparameter tuning.

# Utility files
Located in the `src/bin` directory are various files used to validate, pre-process the base data file into the json file, and demonstrate/experiment with functionality. It contains:
- `check_stable_diffusion_prompts.py` - Script to determine the smallest "correct" stable diffusion prompt's length, then use that (minus some amount) to validate that there are no smaller prompts generated (which would indicate that ChatGPT didn't give us what we wanted). Useful when creating many prompts, so we don't need human interaction to validate.
- `count_usda_categories.py` - Dump out the total counts of various usda categories, including singular vs plural categories
- `create_python_list_of_categories.py` - Simple Python script to read in the USDA Ids and output to standard out text that can be copied/pasted into another Python script as a list.
- `explode_main_data_set.py` - Takes the base dataset and expands it. Will need to be brought back together to for training, but can be merged as needed. Uses ChatGPT to generate StableDiffusion prompts.
- `find_duplicates.py` - A script to count up the duplicate diaries
- `gen_images.py` - Uses a deployed version of [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) to take a stable diffusion prompt and generate an images, saving it to disk via's it's direct API.
- `generate_additional_food_diaries_for_low_category_count.py` - Was previously setup to generate new diaries for under-represented categories, but wasn't used. Instead we decided to drop any category with a total representation of 29 entries or lower.
- `generate_image_to_label_json.py` - Takes information from the exploded directory and creates a json file (above) for model consumption.
- `openai_vision_demo.py` - Proof of concept interaction that reads in a known (in-project) image, sends to ChatGPT
- `stable-diffusion-api-demo.py` - A proof of concept file to test generating images using `stable-diffusion-webui-forge`'s secondary API.
- `update_category_ground_truths.py` - After discussion and Google Doc work, we hard code a mapping of categories to a more generic category or we don't 
- `validate_column_count_of_rows.py` - Sanity check to ensure TSV column counts
- `view_openai_results.py` - A simple web server to show the image, ground truth labels, and predicted labels in a browser, to make viewing the prediction results easy.

