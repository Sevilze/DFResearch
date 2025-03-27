import os
import kagglehub


def get_dataset_path():
    home_dir = os.path.expanduser("~")
    dataset_dir = os.path.join(
        home_dir,
        ".cache",
        "kagglehub",
        "datasets",
        "birdy654",
        "cifake-real-and-ai-generated-synthetic-images",
        "versions",
        "3",
    )
    if not os.path.exists(dataset_dir):
        dataset_dir = kagglehub.dataset_download(
            "birdy654/cifake-real-and-ai-generated-synthetic-images"
        )
    return dataset_dir


def print_directory_tree(root, indent=""):
    for item in os.listdir(root):
        path = os.path.join(root, item)
        print(indent + item)
        if os.path.isdir(path):
            print_directory_tree(path, indent + "  ")


if __name__ == "__main__":
    dataset_path = get_dataset_path()
    print("Path to dataset files:", dataset_path)
