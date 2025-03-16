import kagglehub

def get_dataset_path():
    path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    return path

if __name__ == "__main__":
    dataset_path = get_dataset_path()
    print("Path to dataset files:", dataset_path)
