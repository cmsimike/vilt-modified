import csv
import dotenv
import os

dotenv.load_dotenv()

with open(os.environ["BASE_DATA_FILE"], encoding="utf-8") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    next(tsv_file)  # Skip the header row
    for idx, row in enumerate(tsv_file):
        assert len(row) == 4, f"Row with incorrect column count: {row} on row {idx + 1}"
