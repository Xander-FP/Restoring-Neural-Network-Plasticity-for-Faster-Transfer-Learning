import os
import random
import shutil

def split_dataset(dataset_folder, dest, isRandom=True):
    # Create test and train subfolders
    test_folder = os.path.join(dest, 'test')
    train_folder = os.path.join(dest, 'train')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)

    # Iterate through the dataset folder
    count = 0
    for root, dirs, files in os.walk(dataset_folder):
        for dir in dirs:
            os.makedirs(os.path.join(test_folder, dir), exist_ok=True)
            os.makedirs(os.path.join(train_folder, dir), exist_ok=True)
        for file in files:
            if count % 10 == 0:
                destination_folder = os.path.join(dest, "test")
            else:
                destination_folder = os.path.join(dest, "train")
            count += 1

            # Copy the file to the destination subfolder
            source_path = os.path.join(root, file)
            destination_path = source_path.replace(dataset_folder, destination_folder)
            shutil.copy2(source_path, destination_path)

    print("Dataset split successfully!")

dataset_folder = '.\Datasets\ChestX_min'
destination_folder = '.\Datasets\ChestX_min_split'
split_dataset(dataset_folder, destination_folder)