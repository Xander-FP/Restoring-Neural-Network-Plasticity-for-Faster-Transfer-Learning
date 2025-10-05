import numpy as np
from Data.custom_datasets import CustomFolderDataset, CustomCIFAR10, CustomDTD, CustomFileDataset
from PIL import ImageStat

class DataPrep:
    # This function wraps the datasets such that they can be used by the training data loaders.
    # When adding a new dataset, a new entry should be added to this function
    def get_train_datasets(data_dir, dataset_type):
        # TODO: Let CustomDataset handle CIFAR10 and ImageNet retrieval
        # Where file names are the labels
        if dataset_type == 'File':
            data_dir = data_dir + '/train'
            train_set = CustomFileDataset(data_path=data_dir)
            valid_set = CustomFileDataset(data_path=data_dir) 

        # # Where the folder names are the classes
        if dataset_type == 'Folder':
            data_dir = data_dir + '/train'
            train_set = CustomFolderDataset(data_path=data_dir)
            valid_set = CustomFolderDataset(data_path=data_dir)

        # # # CIFAR10
        if dataset_type == 'Cifar10':
            train_set = CustomCIFAR10(root=data_dir, train=True, download=False)
            valid_set = CustomCIFAR10(root=data_dir, train=True, download=False)

        return train_set, valid_set
    
    # This function wraps the datasets such that they can be used by the training data loaders.
    # When adding a new dataset, a new entry should be added to this function
    def get_test_datasets(data_dir, dataset_type):
        if dataset_type == 'Folder':
            data_dir = data_dir + '/test'
            return CustomFolderDataset(data_path=data_dir)
        
        if dataset_type == 'File':
            data_dir = data_dir + '/test'
            return CustomFileDataset(data_path=data_dir)

        if dataset_type == 'Cifar10':
            return CustomCIFAR10(root=data_dir, train=False, download=False)
    
    def compute_mean_std(data_set, normalize = True):
        rgb_sum = np.array([0, 0, 0], dtype=np.float64)
        rgb_sum_sqrd = np.array([0, 0, 0], dtype=np.float64)
        count = 0
        max_height = 0
        max_width = 0
        for image, _, _ in data_set:
            if image.size[0] > max_width:
                max_width = image.size[0]
            if image.size[1] > max_height:
                max_height = image.size[1]
            stat = ImageStat.Stat(image)
            rgb_sum += np.array(stat.sum)
            rgb_sum_sqrd += np.array(stat.sum2)
            count += stat.count[0]
        rgb_mean = rgb_sum / count
        rgb_stddev = np.sqrt(rgb_sum_sqrd/count - rgb_mean**2)
        if normalize:
            rgb_mean = rgb_mean/256
            rgb_stddev = rgb_stddev/256
        return {'mean': rgb_mean.tolist(), 'std': rgb_stddev.tolist(), 'max_width': max_width, 'max_height': max_height}
    
    # @param total_set: A list of indexes [0,1,2...len(dataset)]
    # This function splits the indexes into a training and validation set
    def split_train_valid_idx(indexes, valid_size = 0.1):
        split = int(np.floor(valid_size * len(indexes)))
        # shuffle the indexes
        np.random.shuffle(indexes)
        train_idx, valid_idx = indexes[split:], indexes[:split]
        return train_idx, valid_idx