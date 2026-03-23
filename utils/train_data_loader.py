import os
import torch
import numpy as np
from monai.data import Dataset
from monai.transforms import LoadImaged, SpatialPadd, RandSpatialCropd, Compose, RandScaleIntensityd, \
    RandShiftIntensityd, Rand3DElasticd, RandAxisFlipd, ToTensord, MapLabelValued, AddChanneld, \
    NormalizeIntensityd, EnsureTyped
from scipy.ndimage import measurements
from monai.transforms import Transform
from monai.config import KeysCollection

class CustomRandomCropd(Transform):
    """
    Custom random crop transform for BraTS data
    """
    def __init__(self, keys: KeysCollection):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        # Generate random crop parameters
        top_crop = np.random.randint(30, 90)
        left_crop = np.random.randint(40, 80)
        z_crop = 16

        # Calculate bottom and right crops
        bottom_crop = top_crop + 128
        right_crop = left_crop + 128
        z_end = z_crop + 128

        d = dict(data)
        for key in self.keys:
            # Assuming input shape is (C, H, W, D)
            current_data = d[key]
            cropped_data = current_data[
                :,  # All channels
                top_crop:bottom_crop,  # Height dimension
                left_crop:right_crop,  # Width dimension
                z_crop:z_end  # Depth dimension
            ]
            d[key] = cropped_data
        return d


class SubjectReader:
    def __init__(self, train_dir, train_dataset, training_size=None):
        """
        Args:
            train_dir: Directory containing training data
            train_dataset: Dataset name (brats2020, brats2021, or brats2023)
            training_size: the size of 3D patch during training
        """
        self.train_dir = train_dir
        self.train_dataset = train_dataset
        self.training_size = training_size

        # Get subject lists
        self.train_subjects = os.listdir(self.train_dir)

    def get_subjects(self, subject_list, data_dir):
        """
        Get subjects from specified directory with dataset-specific file paths
        Args:
            subject_list: list of subject names
            data_dir: directory containing the data
        """
        subjects = []
        for subject_name in subject_list:
            try:
                if self.train_dataset == 'brats2019':
                    subject = {
                        't1': os.path.join(data_dir, subject_name, f'{subject_name}_t1.nii.gz'),
                        't1ce': os.path.join(data_dir, subject_name, f'{subject_name}_t1ce.nii.gz'),
                        't2': os.path.join(data_dir, subject_name, f'{subject_name}_t2.nii.gz'),
                        'flair': os.path.join(data_dir, subject_name, f'{subject_name}_flair.nii.gz'),
                        'label': os.path.join(data_dir, subject_name, f'{subject_name}_seg.nii.gz'),
                        'name': subject_name
                    }
                elif self.train_dataset == 'brats2020':
                    subject = {
                        't1': os.path.join(data_dir, subject_name, f'{subject_name}_t1.nii.gz'),
                        't1ce': os.path.join(data_dir, subject_name, f'{subject_name}_t1ce.nii.gz'),
                        't2': os.path.join(data_dir, subject_name, f'{subject_name}_t2.nii.gz'),
                        'flair': os.path.join(data_dir, subject_name, f'{subject_name}_flair.nii.gz'),
                        'label': os.path.join(data_dir, subject_name, f'{subject_name}_seg.nii.gz'),
                        'name': subject_name
                    }
                elif self.train_dataset == 'brats2021':
                    subject = {
                        't1': os.path.join(data_dir, subject_name, f'{subject_name}-t1n.nii.gz'),
                        't1ce': os.path.join(data_dir, subject_name, f'{subject_name}-t1c.nii.gz'),
                        't2': os.path.join(data_dir, subject_name, f'{subject_name}-t2w.nii.gz'),
                        'flair': os.path.join(data_dir, subject_name, f'{subject_name}-t2f.nii.gz'),
                        'label': os.path.join(data_dir, subject_name, f'{subject_name}-seg.nii.gz'),
                        'name': subject_name
                    }
                elif self.train_dataset == 'brats2023':
                    subject = {
                        't1': os.path.join(data_dir, subject_name, f'{subject_name}-t1n.nii.gz'),
                        't1ce': os.path.join(data_dir, subject_name, f'{subject_name}-t1c.nii.gz'),
                        't2': os.path.join(data_dir, subject_name, f'{subject_name}-t2w.nii.gz'),
                        'flair': os.path.join(data_dir, subject_name, f'{subject_name}-t2f.nii.gz'),
                        'label': os.path.join(data_dir, subject_name, f'{subject_name}-seg.nii.gz'),
                        'name': subject_name
                    }
                else:
                    raise ValueError(f'Unsupported dataset: {self.train_dataset}')

                # Verify that all required files exist
                if all(os.path.exists(path) for path in subject.values() if
                       isinstance(path, str) and path.endswith('.nii.gz')):
                    subjects.append(subject)
                else:
                    print(f"Warning: Missing files for subject {subject_name}, skipping...")

            except Exception as e:
                print(f"Error processing subject {subject_name}: {str(e)}")
                continue

        return subjects


    def get_trainset(self):
        """
        Get the complete training dataset
        """
        train_transform = self.get_training_transform()
        train_subjects = self.get_subjects(self.train_subjects, self.train_dir)
        trainset = Dataset(data=train_subjects, transform=train_transform)
        print(f'Training dataset prepared. Length: {len(trainset)}')
        return trainset

    def get_training_transform(self):
        """
        Data loading and augmentation during training.
        Handles BraTS 2020, 2021, and 2023 datasets with appropriate transformations.
        """
        training_keys = ('t1', 't1ce', 't2', 'flair', 'label')
        image_keys = ('t1', 't1ce', 't2', 'flair')

        transform_list = [
            LoadImaged(keys=training_keys),
            AddChanneld(keys=training_keys),
        ]

        # Add label mapping only for BraTS 2020
        if self.train_dataset in ['brats2019', 'brats2020']:
            transform_list.append(
                MapLabelValued(keys='label', orig_labels=(0, 1, 2, 4), target_labels=(0, 1, 2, 3))
            )

        transform_list.extend([
            SpatialPadd(keys=training_keys, spatial_size=(240, 240, 160)),
            CustomRandomCropd(keys=training_keys),
            NormalizeIntensityd(keys=image_keys),
            RandAxisFlipd(keys=training_keys, prob=0.5),
            RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
            Rand3DElasticd(
                keys=training_keys,
                prob=0.5,
                mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest'),
                sigma_range=(3, 5),
                magnitude_range=(0.1, 0.3),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.05, 0.05, 0.05)
            ),
            EnsureTyped(keys=training_keys)
        ])

        return Compose(transform_list)


