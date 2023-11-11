import fiftyone

def pull_openimages(traning_size, test_size, dataset_dir="openimages"):

    dataset_train = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="train",
        max_samples=traning_size, # 300 000
        label_types=["classifications"],
        dataset_dir=dataset_dir,
    )
    dataset_test = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="test",
        max_samples=test_size, # 10 000
        label_types=["classifications"],
        dataset_dir=dataset_dir,
    )