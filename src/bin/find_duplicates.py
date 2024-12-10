import csv
import dotenv
import os

dotenv.load_dotenv()

diaries = {}
with open(os.environ["BASE_DATA_FILE"], encoding="utf-8") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    next(tsv_file)  # Skip the header row
    for row in tsv_file:
        diary_id = row[0]

        if diary_id not in diaries:
            diaries[diary_id] = 0
        diaries[diary_id] = diaries[diary_id] + 1

# sort the diaries by the number of entries
diaries = dict(sorted(diaries.items(), key=lambda item: item[1], reverse=False))

for diary_id, count in diaries.items():
    if count > 1:
        print(f"Diary {diary_id} has {count} entries")
