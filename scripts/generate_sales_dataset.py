from ml.linear_regression import generate_sales_dataset


if __name__ == "__main__":
    dataset_path = generate_sales_dataset()
    print(f"Generated reproducible synthetic sales dataset: {dataset_path}")
