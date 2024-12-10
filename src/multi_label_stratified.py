import random
from collections import defaultdict, Counter


def multi_label_stratified_split(
    data, test_size=0.2, random_state=None, print_stats=False
):

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0.0 and 1.0")

    if random_state is not None:
        random.seed(random_state)

    # Total number of items
    total_items = len(data)
    num_test = int(round(total_items * test_size))
    num_train = total_items - num_test

    # Collect all labels and counts
    label_counts_total = Counter()
    for _, labels in data:
        label_counts_total.update(labels)

    # Desired counts in train and test sets for each label
    desired_label_counts_test = {
        label: int(round(count * test_size))
        for label, count in label_counts_total.items()
    }
    desired_label_counts_train = {
        label: count - desired_label_counts_test[label]
        for label, count in label_counts_total.items()
    }

    # Initialize counts in train and test sets
    label_counts_test = defaultdict(int)
    label_counts_train = defaultdict(int)

    # Shuffle data
    random.shuffle(data)

    # Initialize train and test sets
    train_data = []
    test_data = []

    for item in data:
        _, labels = item
        # For each item, decide whether to assign to train or test
        # Compute the proportion of desired counts that have been met for each label
        test_proportions = []
        train_proportions = []
        for label in labels:
            test_prop = (
                label_counts_test[label] / desired_label_counts_test[label]
                if desired_label_counts_test[label] > 0
                else 1.0
            )
            train_prop = (
                label_counts_train[label] / desired_label_counts_train[label]
                if desired_label_counts_train[label] > 0
                else 1.0
            )
            test_proportions.append(test_prop)
            train_proportions.append(train_prop)

        # Compute average saturation
        avg_test_saturation = sum(test_proportions) / len(test_proportions)
        avg_train_saturation = sum(train_proportions) / len(train_proportions)

        # Assign to the set where the labels are less saturated
        if avg_test_saturation < avg_train_saturation and len(test_data) < num_test:
            test_data.append(item)
            for label in labels:
                label_counts_test[label] += 1
        else:
            train_data.append(item)
            for label in labels:
                label_counts_train[label] += 1

    # Ensure the test set has the correct number of items
    while len(test_data) < num_test:
        item = train_data.pop()
        test_data.append(item)
        _, labels = item
        for label in labels:
            label_counts_test[label] += 1
            label_counts_train[label] -= 1

    # Ensure the train set has the correct number of items
    while len(train_data) < num_train:
        item = test_data.pop()
        train_data.append(item)
        _, labels = item
        for label in labels:
            label_counts_train[label] += 1
            label_counts_test[label] -= 1

    if print_stats:
        # Report statistics
        print(f"Total items: {total_items}")
        print(f"Train items: {len(train_data)}")
        print(f"Test items: {len(test_data)}")

        # Helper function to print label distribution
        def print_label_distribution(label_counts, dataset_name):
            print(f"\nLabel distribution in {dataset_name}:")
            for label, count in label_counts.items():
                percentage = (
                    count
                    / (
                        len(train_data)
                        if dataset_name == "train set"
                        else (
                            len(test_data)
                            if dataset_name == "test set"
                            else total_items
                        )
                    )
                    * 100
                )
                print(f"{label}: {count} ({percentage:.2f}%)")

        print_label_distribution(label_counts_total, "full dataset")
        print_label_distribution(label_counts_train, "train set")
        print_label_distribution(label_counts_test, "test set")

    # Prepare counts dictionary to return
    counts = {
        "total": dict(label_counts_total),
        "train": dict(label_counts_train),
        "test": dict(label_counts_test),
    }

    return train_data, test_data, counts


# Example/demo/testing which can be invoked by running this script or
# importing the function in another script
if __name__ == "__main__":
    data = [
        ("image1.jpg", ["cat", "dog"]),
        ("image2.jpg", ["cat"]),
        ("image3.jpg", ["dog"]),
        ("image4.jpg", ["rabbit"]),
        ("image5.jpg", ["cat", "rabbit"]),
        ("image6.jpg", ["cat", "dog"]),
        ("image7.jpg", ["dog", "rabbit"]),
        ("image8.jpg", ["rabbit"]),
        ("image9.jpg", ["cat"]),
        ("image10.jpg", ["dog"]),
    ]

    train_data, test_data, counts = multi_label_stratified_split(
        data, test_size=0.2, random_state=42, print_stats=True
    )

    # Dump out the total counts of categories
    print("\nTotal counts of categories:")
    print("Full dataset counts:", counts["total"])
    print("Training set counts:", counts["train"])
    print("Testing set counts:", counts["test"])
