import csv
import dotenv
import os
import pprint

# generated from discussions around the categories in the dataset
updated_categories = {
    "beef": "beef",
    "chicken": "chicken",
    "bread": "bread",
    "lettuce": "lettuce",
    "cheese": "cheese",
    "pork": "pork",
    "spinach": "spinach",
    "egg": "egg",
    "salad dressing": "salad dressing",
    "yogurt": "yogurt",
    "rice": "rice",
    "turkey": "turkey",
    "mushrooms": "mushrooms",
    "peppers": "peppers",
    "orange juice": "juice",
    "milk": "milk",
    "carrots": "carrots",
    "kale": "kale",
    "onions": "onions",
    "corn": "corn",
    "tomatoes": "tomatoes",
    "cereals": "cereals",
    "peas": "peas",
    "muffins": "muffins",
    "cucumber": "cucumber",
    "pineapple": "pineapple",
    "peaches": "peaches",
    "broccoli": "broccoli",
    "cherries": "cherries",
    "waffles": "waffles",
    "coffee": "coffee",
    "pizza": "pizza",
    "blueberries": "blueberries",
    "fish": "fish",
    "strawberries": "strawberries",
    "apples": "apples",
    "cauliflower": "cauliflower",
    "pickles": "pickles",
    "radishes": "radishes",
    "macaroni": "pasta",
    "raspberries": "raspberries",
    "pastries": "pastries",
    "tangerines": "tangerines",
    "celery": "celery",
    "peanut butter": "peanut butter",
    "ice cream": "ice cream",
    "bananas": "bananas",
    "quinoa": "quinoa",
    "chowder": "chowder",
    "kiwifruit": "kiwifruit",
    "popcorn": "popcorn",
    "shrimp": "shrimp",
    "croissants": "croissants",
    "mangos": "mangos",
    "watermelon": "watermelon",
    "potatoes": "potatoes",
    "burrito": "burrito",
    "garlic bread": "garlic bread",
    "grits": "grits",
    "lasagna": "lasagna",
    "sandwich": "sandwich",
    "cake": "cake",
    "ravioli": "ravioli",
    "salmon": "fish",
    "hummus": "hummus",
    "asparagus": "asparagus",
    "baklava": "baklava",
    "beef tartare": "beef tartare",
    "beet salad": "salad",
    "beignets": "beignets",
    "bibimbap": "bibimbap",
    "bread pudding": "bread pudding",
    "bruschetta": "bruschetta",
    "calamari": "calamari",
    "caprese salad": "caprese salad",
    "carrot cake": "carrot cake",
    "ceviche": "ceviche",
    "cheese plate": "cheese",
    "cheese sandwich": "sandwich",
    "cheesecake": "cheesecake",
    "chips": "chips",
    "chips and salsa": "chips",
    "chocolate cake": "chocolate cake",
    "chocolate mousse": "chocolate mousse",
    "churros": "churros",
    "club sandwich": "sandwich",
    "crab cakes": "crab cakes",
    "creme brulee": "creme brulee",
    "croque madame": "croque madame",
    "cup cakes": "cup cakes",
    "deviled egg": "deviled egg",
    "donuts": "donuts",
    "edamame": "edamame",
    "egg benedict": "egg benedict",
    "escargots": "escargots",
    "falafel": "falafel",
    "foie gras": "foie gras",
    "french onion soup": "french onion soup",
    "gnocchi": "gnocchi",
    "greek salad": "salad",
    "gyoza": "gyoza",
    "hot and sour soup": "hot and sour soup",
    "huevos rancheros": "huevos rancheros",
    "jelly": "jelly",
    "lobster bisque": "lobster bisque",
    "lobster sandwich": "sandwich",
    "macarons": "macarons",
    "miso soup": "miso soup",
    "mussels": "mussels",
    "oyster": "oyster",
    "pad thai": "pad thai",
    "paella": "paella",
    "panna cotta": "panna cotta",
    "peking duck": "peking duck",
    "pho": "pho",
    "pork sandwich": "sandwich",
    "poutine": "poutine",
    "ramen": "ramen",
    "risotto": "risotto",
    "samosa": "samosa",
    "sashimi": "sashimi",
    "scallops": "scallops",
    "seaweed salad": "salad",
    "spaghetti bolognese": "pasta",
    "spaghetti carbonara": "pasta",
    "strawberry shortcake": "strawberry shortcake",
    "sushi": "sushi",
    "takoyaki": "takoyaki",
    "triamisu": "triamisu",
    "tuna tartare": "beef tartare",
    "bologna": "bologna",
    "potato salad": "salad",
    "dumplings": "gyoza",
    "hotdog": "hotdog",
    "onion rings": "onion rings",
    "avocados": "avocados",
    "fried rice": "fried rice",
    "arugula": "arugula",
    "bacon": "bacon",
    "cookie": "cookie",
    "oatmeal": "oatmeal",
    "tacos": "tacos",
    "apple pie": "apple pie",
    "guacamole": "guacamole",
    "fruit juice": "juice",
    "potato chips": "chips",
    "spaghetti": "pasta",
    "ham": "ham",
    "nachos": "nachos",
    "melons": "honeydew",
    "honeydrew melons": "honeydew",
    "salami": "salami",
    "tilapia": "fish",
    "tea": "tea",
    "pineapple juice": "juice",
    "pears": "pears",
    "steak": "steak",
    "cookies": "cookies",
    "apricots": "apricots",
    "french fries": "french fries",
    "lemonade": "lemonade",
    "sauce": "sauce",
    "water": "water",
    "cranberry juice": "juice",
    "biscuits": "biscuits",
    "cheese quesadilla": "cheese quesadilla",
    "caesar salad": "salad",
    "egg rolls": "egg rolls",
    "fruit punch": "fruit punch",
    "grape juice": "juice",
    "nuts": "nuts",
    "chicken quesadilla": "chicken quesadilla",
    "plums": "plums",
    "french dressing": "salad dressing",
    "almonds": "almonds",
    "apple juice": "juice",
    "hash brown": "hashbrowns",
    "hash browns": "hashbrowns",
    "smoothie": "smoothie",
    "pasta": "pasta",
    "russian dressing": "salad dressing",
    "blackberries": "blackberries",
    "cheeseburger": "cheeseburger",
}


dotenv.load_dotenv()

base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]


def get_updated_category(categories):
    categories = categories.split(",")

    new_categories = []
    for category in categories:
        category = category.strip()
        if category in updated_categories:
            new_categories.append(updated_categories[category])

    return new_categories


with os.scandir(base_directory) as entries:
    for entry in entries:
        if entry.is_dir():
            subdir = os.path.join(base_directory, entry.name)
            original_ground_truth_labels_file = os.path.join(
                subdir, "ground_truth_labels.txt"
            )
            if not os.path.exists(original_ground_truth_labels_file):
                print(f"Skipping {subdir} as ground truth labels do not exist")
                continue
            with open(original_ground_truth_labels_file, "r") as f:
                original_ground_truth_labels = f.read()

            result = get_updated_category(original_ground_truth_labels)
            if len(result) == 0:
                continue

            new_ground_truth_labels_file = os.path.join(
                subdir, "ground_truth_labels_updated.txt"
            )

            new_ground_truth_labels = ",".join(result)
            with open(new_ground_truth_labels_file, "w") as f:
                f.write(new_ground_truth_labels)
