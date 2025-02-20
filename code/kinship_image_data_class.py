
import pandas as pd
import random
from pathlib import Path
from typing import List, Tuple, Optional
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from itertools import product
from sklearn.model_selection import train_test_split
import itertools
import numpy as np

# Set the seed value
SEED = 42
# Set the seed for Python's random module
random.seed(SEED)
# Set the seed for NumPy
np.random.seed(SEED)
# Set the seed for PyTorch
torch.manual_seed(SEED)


class KinshipPairs(Dataset):
    """
    This class allows to easly work with the Dataset and Dataloader class in PyTorch to preform training,testing and evaluations.
    """

    def __init__(self, pairs, main_image_dir_path):
        """
        :param pairs: List of tuples (img1_path, img2_path, label). (Label = 1 => Blood Related / Label = 0 => Not Blood Related)
        :param families_dir: Base directory containing family images.
        """
        self.pairs = pairs
        self.main_image_dir_path = main_image_dir_path

        # self.transform = transforms.Compose([
        #     transforms.Resize((112, 112)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        # ])

        # Image transformation: Resize and normalize the image as per Facenet input requirements
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Resize to match input size for InceptionResnetV1
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on pre-trained model stats
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        full_img1_path = os.path.join(self.main_image_dir_path, img1_path)
        full_img2_path = os.path.join(self.main_image_dir_path, img2_path)

        img1 = Image.open(full_img1_path).convert("RGB")
        img2 = Image.open(full_img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32), full_img1_path, full_img2_path



class KinshipDataset:
    def __init__(self, train_pairs_dir: str, train_families_dir: str, test_faces_dir: str, test_relationship_lists_dir: str, test_relationship_labels_dir: str, train_ratio: float, validation_ratio: float, test_ratio: float, balanced_classes: bool):
        """
        :param train_pairs_dir: Directory containing CSV files with training pairs.
        :param train_families_dir: Directory containing family image data for training.
        :param test_faces_dir: Directory containing all test images.
        :param test_relationship_lists_dir: Directory containing CSV files with test pairs.
        :param test_relationship_labels_dir: Directory containing CSV files with test labels.
        :param validation_ratio: Ratio of training families to be used for validation.
        :param balanced_classes: Weather to limit negative pair sampels to be equal to the number of positive pairs.
        """
        # Directory Paths
        self.train_pairs_dir = Path(train_pairs_dir)
        self.train_families_dir = Path(train_families_dir)
        self.test_faces_dir = Path(test_faces_dir)

        # Origial Test Data
        self.test_relationship_lists_dir = Path(test_relationship_lists_dir)
        self.test_relationship_labels_dir = Path(test_relationship_labels_dir)
        self.original_test_pairs = self._load_test_pairs() # 

        # Original Train Data (Split to train, validation and test.)
        self.train_pair_types = self._load_pair_types(self.train_pairs_dir)
        self.train_pairs, self.validation_pairs, self.test_pairs = self._load_pairs(train_ratio=train_ratio, validation_ratio=validation_ratio, test_ratio=test_ratio, balanced_classes=balanced_classes)


    def _load_pair_types(self, pairs_dir):
        pair_files = pairs_dir.glob("*.csv")
        pair_types = {}
        for file in pair_files:
            pair_name = file.stem
            pair_df = pd.read_csv(file)
            pair_types[pair_name] = pair_df
        return pair_types

    def _load_family_data(self, families_dir):
        """
        Create a directory with relative paths to images of each member in each family directory from the Training set
        :param families_dir: path to directory containing directories of families, each with members, each of which contain images
        :return:
        """
        family_data = {}
        for family in families_dir.iterdir():
            if family.is_dir():
                family_id = family.name
                member_data = {}
                for member in family.iterdir():
                    if member.is_dir():
                        member_id = member.name
                        image_paths = list(member.glob("*.jpg"))

                        # Turn full paths into relative paths
                        image_relative_paths = []
                        for image_path in image_paths:
                            relative_path = "\\".join(str(image_path).split("\\")[-3:])
                            image_relative_paths.append(relative_path)
                        member_data[member_id] = image_relative_paths

                family_data[family_id] = member_data

        return family_data

    def _load_test_pairs(self):
        """
        Load test pairs and their labels from the test directories.
        :return: List of tuples (img1_path, img2_path, label).
        """
        pairs = []
        list_files = self.test_relationship_lists_dir.glob("*.csv")
        for list_file in list_files:
            label_file = self.test_relationship_labels_dir / list_file.name
            if not label_file.exists():
                raise FileNotFoundError(f"Label file {label_file} not found for list file {list_file}")

            list_df = pd.read_csv(list_file)
            label_df = pd.read_csv(label_file)

            if len(list_df) != len(label_df):
                raise ValueError(f"Mismatch in number of rows between {list_file} and {label_file}")

            for (_, row_list), (_, row_label) in zip(list_df.iterrows(), label_df.iterrows()):
                img1_path = row_list["p1"]
                img2_path = row_list["p2"]
                label = int(row_label["label"])
                pairs.append((img1_path, img2_path, label))

        random.shuffle(pairs)

        return pairs

    def _load_pairs(self, train_ratio: float, validation_ratio: float, test_ratio: float, balanced_classes: bool):
        # Ensure the ratios sum to 1
        if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and 0 < test_ratio < 1):
            raise ValueError("All ratios must be between 0 and 1.")
        if not abs((train_ratio + validation_ratio + test_ratio) - 1.0) < 1e-6:
            raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1.")
        
        # Load families data dict
        family_data_dict = self._load_family_data(self.train_families_dir)
        
        ################ Split to Train, Validation & Test Families Dicts ################
        keys = list(family_data_dict.keys())
        random.seed(42)
        random.shuffle(keys)

        # Calculate split ranges
        train_split = int(len(keys) * train_ratio)
        validation_split = train_split + int(len(keys) * validation_ratio)

        # Select familiy ids (keys) to be in training/validation/test
        train_keys = keys[:train_split]
        validation_keys = keys[train_split:validation_split]
        test_keys = keys[validation_split:]

        # Save the family dicts with data for each split
        train_families_dict = {k: family_data_dict[k] for k in train_keys}
        validation_families_dict = {k: family_data_dict[k] for k in validation_keys}
        test_families_dict = {k: family_data_dict[k] for k in test_keys}
        
        self.train_families_data_dict = train_families_dict
        self.validation_families_data_dict = validation_families_dict
        self.test_families_data_dict = test_families_dict
        
        ################ Generate Positive & Negative Pairs ################
        train_positive_pairs, validation_positive_pairs, test_positive_pairs = self._generate_positive_pairs(self.train_pair_types)
        train_negative_pairs = self._generate_negative_pairs(self.train_families_data_dict, n=len(train_positive_pairs) if balanced_classes else 0)
        validation_negative_pairs = self._generate_negative_pairs(self.validation_families_data_dict, n=len(validation_positive_pairs) if balanced_classes else 0)
        test_negative_pairs = self._generate_negative_pairs(self.test_families_data_dict, n=len(test_positive_pairs) if balanced_classes else 0)
        
        train_pairs = train_positive_pairs + train_negative_pairs
        validation_pairs = validation_positive_pairs + validation_negative_pairs
        test_pairs = test_positive_pairs + test_negative_pairs

        random.seed(42)
        random.shuffle(train_pairs)
        random.shuffle(validation_pairs)
        random.shuffle(test_pairs)
        
        return train_pairs, validation_pairs, test_pairs

    def _generate_positive_pairs(self, pair_types):
        train_families_set = set(self.train_families_data_dict.keys())
        validation_families_set = set(self.validation_families_data_dict.keys())
        test_families_set = set(self.test_families_data_dict.keys())
    
        train_pairs = []
        validation_pairs = []
        test_pairs = []
    
        for pair_type, pair_df in pair_types.items():
            for _, row in pair_df.iterrows():
                family = row['p1'].split("/")[0]
                member_1_dir_path = os.path.join(self.train_families_dir, row["p1"])
                member_2_dir_path = os.path.join(self.train_families_dir, row["p2"])
    
                # List all image files in the directories
                member_1_images = [os.path.join(member_1_dir_path, img).split("\\")[-1] for img in os.listdir(member_1_dir_path)]
                member_2_images = [os.path.join(member_2_dir_path, img).split("\\")[-1] for img in os.listdir(member_2_dir_path)]
    
                # Create all combinations of images between member 1 and member 2
                pairs = [(img1, img2, 1) for img1, img2 in itertools.product(member_1_images, member_2_images)]
    
                if family in train_families_set:
                    train_pairs.extend(pairs)
                elif family in validation_families_set:
                    validation_pairs.extend(pairs)
                elif family in test_families_set:
                    test_pairs.extend(pairs)
        
        return train_pairs, validation_pairs, test_pairs


    def _generate_negative_pairs(self, family_data, n=0):
        pairs = []
        family_member_pairs = [
            (family_id, member_id)
            for family_id, members in family_data.items()
            for member_id in members.keys()
        ]

        # Generate all combinations of family member pairs
        all_family_pairs = list(product(family_member_pairs, repeat=2))

        # Create a subsample of size n
        random.seed(42)
        random.shuffle(all_family_pairs)
        sampled_family_pairs = []
        for (family_1, member_1), (family_2, member_2) in all_family_pairs:
            if family_1 != family_2 and family_1 <= family_2 and len(sampled_family_pairs)<n:
                sampled_family_pairs.append(((family_1, member_1), (family_2, member_2)))
            elif len(sampled_family_pairs) == n:
                break


        for (family_1, member_1), (family_2, member_2) in sampled_family_pairs:

            person_1_relative_path = f"{family_1}/{member_1}/"
            person_2_relative_path = f"{family_2}/{member_2}/"
            person_1_dir_path = os.path.join(self.train_families_dir, person_1_relative_path)
            person_2_dir_path = os.path.join(self.train_families_dir, person_2_relative_path)

            # List all image files in the directories
            person_1_images = [os.path.join(person_1_relative_path, img) for img in os.listdir(person_1_dir_path)]
            person_2_images = [os.path.join(person_2_relative_path, img) for img in os.listdir(person_2_dir_path)]

            # Randomly choose one combination of images between member 1 and member 2
            if person_1_images and person_2_images:
                img1 = random.choice(person_1_images)
                img2 = random.choice(person_2_images)
                pairs.append((img1, img2, 0))  # Label 0 for negative pair

        # pairs = list(set(pairs))
        selected_negative_pairs = pairs

        return selected_negative_pairs

    def get_statistics(self, pairs):
        """
        Computes statistics for the given pairs and family data.
        :param pairs: List of pairs (img1_path, img2_path, label).
        :param family_data: Dictionary of family data (optional, for training data).
        :return: Dictionary with statistics.
        """
        positive_count = sum(1 for _, _, label in pairs if label == 1)
        negative_count = sum(1 for _, _, label in pairs if label == 0)

        stats = {
            "positive_pairs": positive_count,
            "negative_pairs": negative_count,
            "total_pairs": len(pairs),
        }

        return stats




if __name__ == '__main__':
    # Example usage:
    train_pairs_dir = r"D:\University\year 4\semester_1\Big Data\Project\Data\families_in_the_wild\train\train-relationship-lists"
    train_families_dir = r"D:\University\year 4\semester_1\Big Data\Project\Data\families_in_the_wild\train\train-faces"

    test_relationship_lists_dir = r"D:\University\year 4\semester_1\Big Data\Project\Data\families_in_the_wild\test\test-relationship-lists"
    test_faces_dir = r"D:\University\year 4\semester_1\Big Data\Project\Data\families_in_the_wild\test\test-faces"
    test_relationship_labels_dir = r"D:\University\year 4\semester_1\Big Data\Project\Data\families_in_the_wild\test\test-relationship-labels"
    dataset = KinshipDataset(train_pairs_dir, train_families_dir,test_faces_dir,test_relationship_lists_dir,test_relationship_labels_dir)

    train_pairs, validation_pairs, test_pairs = dataset.train_pairs, dataset.validation_pairs, dataset.test_pairs

    train_dataset = KinshipPairs(train_pairs, train_families_dir)
    validation_dataset = KinshipPairs(validation_pairs, train_families_dir)
    test_dataset = KinshipPairs(test_pairs, test_faces_dir)


    train_stats = dataset.get_statistics(train_pairs, family_data=dataset.train_families_data_dict)
    validation_stats = dataset.get_statistics(validation_pairs, family_data=dataset.train_families_data_dict)
    test_stats = dataset.get_statistics(test_pairs)

    print("\nTrain Statistics:", train_stats)
    print("\nValidation Statistics:", validation_stats)
    print("\nTest Statistics:", test_stats)