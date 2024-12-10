import csv
import dotenv
import os
import pprint

dotenv.load_dotenv()
category_count = {}


# This is needed because pprint does not preserve the order of the dictionary
def pretty_print_ordered_dict(d):
    for i, (key, value) in enumerate(d.items()):
        print(f"{key}: {value}")


with open(os.environ["BASE_DATA_FILE"], encoding="utf-8") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    next(tsv_file)  # Skip the header row
    for row in tsv_file:
        category_tsv = row[1]
        categories = category_tsv.split(",")
        # trim the categories
        categories = [category.strip() for category in categories]
        for category in categories:
            if category not in category_count:
                category_count[category] = 0
            category_count[category] += 1

# sort the dictionary by category count
category_count = dict(sorted(category_count.items(), key=lambda x: x[1], reverse=True))
pretty_print_ordered_dict(category_count)
print(len(category_count.keys()))

# Dump out the list of keys formatted as a Python list
keys_list = list(category_count.keys())
formatted_keys = repr(keys_list)
# prints out the list of keys in a format that can be copied and pasted into a Python script
print("\nFormatted list of keys:")
print(formatted_keys)

# print the map to stdout in a format that can be copied and pasted into a Python script
print("\nFormatted dictionary:")
pprint.pprint(category_count)
