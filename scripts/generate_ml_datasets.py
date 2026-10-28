from ml.datasets import generate_all_datasets


if __name__ == "__main__":
    for dataset_path in generate_all_datasets():
        print(f"Generated synthetic dataset: {dataset_path}")
