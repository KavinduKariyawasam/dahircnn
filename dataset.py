import os
from pathlib import Path
import json
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box, MultiPolygon
from shapely import affinity
from shapely.wkt import loads

import rasterio
from rasterio.mask import mask as rio_mask

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import functional as TF

from tqdm.notebook import tqdm

# Visualization (optional)
import matplotlib.pyplot as plt

# Constants and Configuration
MEAN_TRAIN = torch.tensor([0.3052, 0.3381, 0.2510, 0.3073, 0.3427, 0.2542])
STD_TRAIN = torch.tensor([0.1630, 0.1449, 0.1360, 0.1609, 0.1426, 0.1348])

MEAN_TEST = torch.tensor([0.1395, 0.1422, 0.1433, 0.1370, 0.1298, 0.1318])
STD_TEST = torch.tensor([0.2325, 0.2271, 0.2217, 0.2182, 0.2048, 0.2003])

DATA_DIR = "/home/deependra/Dataset"

# Helper Functions
def normalize(image_tensor, mean, std):
    return TF.normalize(image_tensor, mean, std)

def parse_coords(coord_str):
    """
    Extract latitude and longitude from a coordinate string.
    Example format: "(lat;lon)"
    """
    coord_str = coord_str.strip("()")
    lat_str, lon_str = coord_str.split(";")
    lat = float(lat_str)
    lon = float(lon_str)
    return lat, lon

def get_subtype_counts(label_path):
    """
    Count the occurrences of each subtype in the label JSON.
    """
    with open(label_path, "r") as file:
        label_data = json.load(file)

    subtypes = [
        building["properties"]["subtype"] for building in label_data["features"]["xy"]
    ]
    return Counter(subtypes)

def mask_pixels_outside_rectangle(image, top, left, bottom, right):
    """
    Zero out (black out) all pixels outside the [top:bottom, left:right] rectangle
    in pixel coordinates.

    Args:
        image (np.ndarray): Image array of shape (C, H, W).
        top (int): Top pixel coordinate.
        left (int): Left pixel coordinate.
        bottom (int): Bottom pixel coordinate.
        right (int): Right pixel coordinate.

    Returns:
        np.ndarray: Masked image array.
    """
    channels, height, width = image.shape

    # Clamp corners to valid boundaries
    top = max(0, min(top, height))
    left = max(0, min(left, width))
    bottom = max(0, min(bottom, height))
    right = max(0, min(right, width))

    masked = np.copy(image)
    # Top strip
    if top > 0:
        masked[:, :top, :] = 0
    # Bottom strip
    if bottom < height:
        masked[:, bottom:, :] = 0
    # Left strip
    if left > 0:
        masked[:, :, :left] = 0
    # Right strip
    if right < width:
        masked[:, :, right:] = 0

    return masked

def clip_polygon_to_rectangle(polygon, top, left, bottom, right):
    """
    Intersect 'polygon' with the rectangle (left, top, right, bottom).

    Args:
        polygon (shapely.geometry.Polygon): Polygon to clip.
        top (int): Top pixel coordinate.
        left (int): Left pixel coordinate.
        bottom (int): Bottom pixel coordinate.
        right (int): Right pixel coordinate.

    Returns:
        shapely.geometry.Polygon or MultiPolygon: Clipped polygon.
    """
    rect = box(left, top, right, bottom)
    return polygon.intersection(rect)

def polygon_to_mask(polygon, width, height):
    """
    Rasterize a single polygon into a binary mask of shape [height, width].

    Args:
        polygon (shapely.geometry.Polygon or MultiPolygon): Polygon to rasterize.
        width (int): Width of the mask.
        height (int): Height of the mask.

    Returns:
        np.ndarray: Binary mask array.
    """
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if polygon.is_empty:
        return np.array(mask_img, dtype=np.uint8)

    # Handle Polygon vs MultiPolygon
    if polygon.geom_type == "Polygon":
        if polygon.exterior is not None:
            x, y = polygon.exterior.coords.xy
            coords = [(xi, yi) for xi, yi in zip(x, y)]
            draw.polygon(coords, outline=1, fill=1)
    elif polygon.geom_type == "MultiPolygon":
        for poly in polygon.geoms:
            if poly.exterior is not None:
                x, y = poly.exterior.coords.xy
                coords = [(xi, yi) for xi, yi in zip(x, y)]
                draw.polygon(coords, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)

# Enhanced Resize Transform
class ResizeTransform:
    """
    Resizes the image and adjusts the bounding boxes and masks accordingly.
    """

    def __init__(self, size):
        """
        Args:
            size (tuple): Desired output size as (height, width).
        """
        self.size = size  # (height, width)

    def __call__(self, image, target):
        """
        Apply the resize transform.

        Args:
            image (PIL.Image): Image to resize.
            target (dict): Dictionary containing 'boxes' and 'masks'.

        Returns:
            tuple: Resized image and updated target.
        """
        orig_width, orig_height = image.size  # PIL Image size: (width, height)
        new_height, new_width = self.size      # Desired size: (height, width)

        # Perform the resize
        image = TF.resize(image, self.size)

        # Calculate scaling ratios correctly
        ratio_width  = new_width / orig_width
        ratio_height = new_height / orig_height

        # Adjust bounding boxes
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"]
            # boxes are [xmin, ymin, xmax, ymax]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_height
            target["boxes"] = boxes

        # Adjust masks
        if "masks" in target and target["masks"].numel() > 0:
            masks = target["masks"]  # shape: (N, H, W)
            masks = masks.unsqueeze(1)  # (N,1,H,W)
            masks = TF.resize(masks, self.size, interpolation=TF.InterpolationMode.NEAREST)
            masks = masks.squeeze(1)  # back to (N,H,W)
            target["masks"] = masks

        return image, target

# Updated BuildingDataset with Augmentations
class BuildingDataset(Dataset):
    """
    A PyTorch Dataset class for training models on building images.
    Handles both pre-disaster and post-disaster images with optional augmentations.
    """

    def __init__(self, df, resize_size=(512, 512), augment=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image and label paths.
            resize_size (tuple): Desired output size as (height, width).
            augment (bool): Whether to apply augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.resize_size = resize_size
        self.augment = augment
        self.damage_class_to_id = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
        }
        self.resize_transform = ResizeTransform(self.resize_size)

        # Define augmentations (example: Color Jitter)
        self.augmentations = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves and processes an image-target pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: {
                "img": Tensor(C, H, W),
                "target": dict containing 'boxes', 'labels', 'masks', 'image_id',
                "counts": dict with damage counts
            }
        """
        row = self.df.iloc[idx]
        img_path = row.post_image_path
        img_pre_path = row.pre_image_path
        label_path = row.post_label_path
        image_id = torch.tensor([idx])
        damage_counts = {
            "no-damage": row.get("no-damage", 0),
            "minor-damage": row.get("minor-damage", 0),
            "major-damage": row.get("major-damage", 0),
            "destroyed": row.get("destroyed", 0),
        }

        try:
            # Load pre-disaster image
            with rasterio.open(img_pre_path) as src_pre:
                img_pre_array = src_pre.read()
                if img_pre_array.shape[0] >= 3:
                    img_pre_array = img_pre_array[:3, :, :]
                else:
                    img_pre_array = np.repeat(img_pre_array, 3, axis=0)

                img_pre_array = np.transpose(img_pre_array, (1, 2, 0)).astype(np.uint8)
                img_pre = Image.fromarray(img_pre_array)

            # Load post-disaster image
            with rasterio.open(img_path) as src_post:
                img_post_array = src_post.read()
                if img_post_array.shape[0] >= 3:
                    img_post_array = img_post_array[:3, :, :]
                else:
                    img_post_array = np.repeat(img_post_array, 3, axis=0)

                img_post_array = np.transpose(img_post_array, (1, 2, 0)).astype(np.uint8)
                img = Image.fromarray(img_post_array)
        except Exception as e:
            print(f"Error reading image: {img_path}")
            print(e)
            return {
                "img": None,
                "target": {'image_id': image_id},
                "counts": None
            }

        width, height = img.size

        # Load annotations
        with open(label_path, "r") as f:
            annotations = json.load(f)

        boxes = []
        labels = []
        masks = []
        annotations = annotations["features"]["xy"]

        for annotation in annotations:
            properties = annotation["properties"]
            subtype = properties.get("subtype", "no-damage")
            damage_label = self.damage_class_to_id.get(subtype, 0)

            polygon_wkt = annotation["wkt"]
            polygon = loads(polygon_wkt)

            if not polygon.is_valid or polygon.is_empty:
                continue

            # Clip polygon to image boundaries
            polygon = self._clip_polygon_to_image(polygon, width, height)
            if polygon.is_empty:
                continue

            xmin, ymin, xmax, ymax = polygon.bounds
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(damage_label)

            mask = self._polygon_to_mask(polygon, width, height)
            masks.append(mask)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8),
                "image_id": image_id,
            }
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": image_id,
            }

        # Apply Resize Transform
        img_pre, _ = self.resize_transform(img_pre, {})
        img, target = self.resize_transform(img, target)

        # Apply augmentations if enabled
        if self.augment:
            # Concatenate pre and post images for joint augmentation
            imgs_combined = Image.merge('RGB', (
                img.convert('RGB'),
                img_pre.convert('RGB'),
                img.convert('RGB')  # Example: Repeat post image channels for demonstration
            ))

            imgs_combined = self.augmentations(imgs_combined)
            # Split the images back
            img = imgs_combined.getchannel(0)
            img_pre = imgs_combined.getchannel(1)

        # Convert images to tensors
        img_pre = TF.to_tensor(img_pre)
        img = TF.to_tensor(img)

        # Normalize images (optional, uncomment if needed)
        # img_pre = normalize(img_pre, MEAN_TRAIN[3:], STD_TRAIN[3:])
        # img = normalize(img, MEAN_TRAIN[:3], STD_TRAIN[:3])

        # Concatenate pre and post images along the channel dimension
        imgs = torch.cat([img, img_pre], dim=0)

        return {
            "img": imgs,
            "target": target,
            "counts": damage_counts
        }

    def _polygon_to_mask(self, polygon, width, height):
        """
        Converts a polygon to a binary mask.

        Args:
            polygon (shapely.geometry.Polygon): Polygon to convert.
            width (int): Width of the mask.
            height (int): Height of the mask.

        Returns:
            np.ndarray: Binary mask array.
        """
        return polygon_to_mask(polygon, width, height)

    def _clip_polygon_to_image(self, polygon, width, height):
        """
        Clips a polygon to the image boundaries.

        Args:
            polygon (shapely.geometry.Polygon): Polygon to clip.
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            shapely.geometry.Polygon or MultiPolygon: Clipped polygon.
        """
        image_box = box(0, 0, width, height)
        return polygon.intersection(image_box)

# TestDataset Class (unchanged)
class TestDataset(Dataset):
    """
    A PyTorch Dataset class for test images of buildings.
    """

    def __init__(self, df, resize_size=(512, 512)):
        """
        Args:
            df (pd.DataFrame): DataFrame containing test image paths and metadata.
            resize_size (tuple): Desired output size as (height, width).
        """
        self.df = df.reset_index(drop=True)
        self.resize_size = resize_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.post_image_path
        img_pre_path = row.pre_image_path
        img_id = row.id

        try:
            # Load pre-disaster image
            with rasterio.open(img_pre_path) as src_pre:
                img_pre_array = src_pre.read()
                if img_pre_array.shape[0] >= 3:
                    img_pre_array = img_pre_array[:3, :, :]
                else:
                    img_pre_array = np.repeat(img_pre_array, 3, axis=0)

                img_pre_array = np.transpose(img_pre_array, (1, 2, 0)).astype(np.uint8)
                img_pre = Image.fromarray(img_pre_array)

            # Load post-disaster image
            with rasterio.open(img_path) as src_post:
                img_post_array = src_post.read()
                if img_post_array.shape[0] >= 3:
                    img_post_array = img_post_array[:3, :, :]
                else:
                    img_post_array = np.repeat(img_post_array, 3, axis=0)

                img_post_array = np.transpose(img_post_array, (1, 2, 0)).astype(np.uint8)
                img = Image.fromarray(img_post_array)
        except Exception as e:
            print(f"Error reading image: {img_path}")
            print(e)
            return {
                "img": None,
                "img_id": img_id
            }

        # Resize images
        img = TF.resize(img, self.resize_size)
        img = TF.to_tensor(img)

        img_pre = TF.resize(img_pre, self.resize_size)
        img_pre = TF.to_tensor(img_pre)

        # Normalize images (optional, uncomment if needed)
        # img = normalize(img, MEAN_TEST[:3], STD_TEST[:3])
        # img_pre = normalize(img_pre, MEAN_TEST[3:], STD_TEST[3:])

        # Concatenate pre and post images along the channel dimension
        imgs = torch.cat([img, img_pre], dim=0)

        return {
            "img": imgs,
            "img_id": img_id
        }

# Collate Functions
def collate_fn(batch):
    """
    Custom collate function for training DataLoader.
    """
    imgs = [item["img"] for item in batch]
    targets = [item["target"] for item in batch]
    counts = [item["counts"] for item in batch]
    return imgs, targets, counts

def test_collate_fn(batch):
    """
    Custom collate function for test DataLoader.
    """
    imgs = [item["img"] for item in batch]
    img_ids = [item["img_id"] for item in batch]
    return imgs, img_ids

# DataFrame Creation Functions
def test_df_create():
    """
    Creates the test DataFrame by merging image paths with coordinates.
    """
    # Load test image coords and image paths
    test_coords = pd.read_csv(f"{DATA_DIR}/malawi_test/test_image_coords.csv")
    test_pre_df = pd.Series(
        [
            x
            for x in Path(f"{DATA_DIR}/malawi_test/Images/").glob("*.tif")
            if x.parts[-1].split(".")[0].split("_")[-2] == "pre"
        ]
    ).to_frame(name="pre_image_path")
    test_post_df = pd.Series(
        [
            x
            for x in Path(f"{DATA_DIR}/malawi_test/Images/").glob("*.tif")
            if x.parts[-1].split(".")[0].split("_")[-2] == "post"
        ]
    ).to_frame(name="post_image_path")

    test_pre_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in test_pre_df.pre_image_path
    ]
    test_post_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in test_post_df.post_image_path
    ]

    test_df = test_pre_df.merge(test_post_df, how="left", on="id")
    test_df = test_coords.merge(test_df, how="left", on="id")
    test_df["flood_name"] = ["_".join(x.split("_")[:-1]) for x in test_df.id]

    # Add columns for min/max lat/lon
    test_df["post_min_lon"] = None
    test_df["post_max_lon"] = None
    test_df["post_min_lat"] = None
    test_df["post_max_lat"] = None

    for idx, row in test_df.iterrows():
        # Parse all four corner coordinates
        corners = [
            parse_coords(row["pre_top_left"]),
            parse_coords(row["pre_top_right"]),
            parse_coords(row["pre_bottom_right"]),
            parse_coords(row["pre_bottom_left"]),
        ]

        # Extract all latitudes and longitudes into separate lists
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]

        # Compute min/max latitude and longitude
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)

        # Assign to DataFrame
        test_df.at[idx, "post_min_lat"] = min_lat
        test_df.at[idx, "post_max_lat"] = max_lat
        test_df.at[idx, "post_min_lon"] = min_lon
        test_df.at[idx, "post_max_lon"] = max_lon

    test_df.to_csv("/home/deependra/Dataset/malawi_test/test_df.csv", index=False)
    return test_df

def train_df_create(split="hold"):
    """
    Creates the training DataFrame by merging image paths with labels.
    """
    dfs = []
    image_dir = Path(f"{DATA_DIR}/x2view/geotiffs/{split}/images")
    label_dir = Path(f"{DATA_DIR}/x2view/geotiffs/{split}/labels")

    image_paths = list(image_dir.rglob("*.tif"))
    images_df = pd.DataFrame(image_paths, columns=["image_path"])

    images_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in images_df.image_path
    ]
    images_df["pre_post"] = [
        x.parts[-1].split(".")[0].split("_")[-2] for x in images_df.image_path
    ]

    label_paths = list(label_dir.rglob("*.json"))
    labels_df = pd.DataFrame(label_paths, columns=["label_path"])

    labels_df["id"] = [
        "_".join(x.parts[-1].split(".")[0].split("_")[:-2])
        for x in labels_df.label_path
    ]
    labels_df["pre_post"] = [
        x.parts[-1].split(".")[0].split("_")[-2] for x in labels_df.label_path
    ]

    merged_df = images_df.merge(labels_df, how="left", on=["id", "pre_post"])
    merged_df = merged_df[["id", "pre_post", "image_path", "label_path"]]

    pre_df = merged_df[merged_df.pre_post == "pre"]
    post_df = merged_df[merged_df.pre_post == "post"]

    pre_df.columns = [f"pre_{x.replace('-', '_')}" for x in pre_df.columns]
    pre_df = pre_df.drop(["pre_pre_post"], axis=1)
    pre_df = pre_df.rename(columns={"pre_id": "id"})

    post_df.columns = [f"post_{x.replace('-', '_')}" for x in post_df.columns]
    post_df = post_df.drop(["post_pre_post"], axis=1)
    post_df = post_df.rename(
        columns={"post_id": "id"}
    )

    merged_df = pre_df.merge(post_df, how="left", on=["id"])

    dfs.append(merged_df)

    df = pd.concat(dfs, ignore_index=True)
    df["flood_name"] = ["_".join(x.split("_")[:-1]) for x in df.id]

    # Process each label_path and aggregate counts
    all_counts = []
    for label_path in df["post_label_path"]:
        subtype_counts = get_subtype_counts(label_path)
        all_counts.append(subtype_counts)

    # Convert counts to DataFrame and combine with original DataFrame
    counts_df = (
        pd.DataFrame(all_counts).fillna(0).astype(int)
    )  # Fill NaN with 0 and cast to int
    df = pd.concat([df, counts_df], axis=1)

    # Drop un-classified, as the main target categories include no-damage, minor-damage, major-damage, destroyed
    if "un-classified" in df.columns:
        df = df.drop("un-classified", axis=1)

    df.to_csv(f"/home/deependra/Dataset/malawi_test/train_df_{split}.csv", index=False)
    return df

# Combined Dataset Class for Augmentation (Optional)
class CombinedDataset(Dataset):
    """
    A dataset that combines two datasets.
    It alternates between samples from dataset1 and dataset2.
    """

    def __init__(self, dataset1, dataset2):
        """
        Args:
            dataset1 (Dataset): First dataset.
            dataset2 (Dataset): Second dataset.
        """
        assert len(dataset1) == len(dataset2), "Datasets must have the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(dataset1) + len(dataset2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            # Get sample from dataset1
            sample = self.dataset1[idx]
            # Ensure the output structure matches
            imgs = [sample["img"]]
            targets = [sample["target"]]
        else:
            # Get sample from dataset2
            sample = self.dataset2[idx - len(self.dataset1)]
            # sample contains pre and post images
            imgs = [sample["pre_img"], sample["post_img"]]
            targets = [sample["pre_target"], sample["post_target"]]
        return imgs, targets

# Combined Collate Function (Optional)
def collate_fn_combined(batch):
    """
    Custom collate function for the CombinedDataset.
    """
    imgs = []
    targets = []
    for imgs_list, targets_list in batch:
        imgs.extend(imgs_list)
        targets.extend(targets_list)
    return imgs, targets

# Main Execution Block
if __name__ == "__main__":

    # Create or load test DataFrame
    test_df_path = "/home/deependra/Dataset/malawi_test/test_df.csv"
    if os.path.exists(test_df_path):
        print("Loading test_df from file")
        test_df = pd.read_csv(test_df_path)
    else:
        print("Creating test_df")
        test_df = test_df_create()

    # Instantiate TestDataset and DataLoader
    test_dataset = TestDataset(test_df, resize_size=(512, 512))
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=test_collate_fn,
    )

    print("Test data loading complete")

    # Create or load training DataFrame
    train_df_path = "/home/deependra/Dataset/malawi_test/train_df_tier1.csv"
    if os.path.exists(train_df_path):
        print("Loading train_df from file")
        train_df = pd.read_csv(train_df_path)
    else:
        print("Creating train_df")
        train_df = train_df_create(split="tier1")

    # Instantiate BuildingDataset with augmentations
    train_dataset = BuildingDataset(
        train_df,
        resize_size=(512, 512),
        augment=True  # Set to True to enable augmentations
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )

    print("Train data loading complete")

    # Example Loop (for debugging)
    for i, data in enumerate(train_loader):
        imgs, targets, counts = data
        print(f"Batch {i+1}:")
        print(f"Number of images in batch: {len(imgs)}")
        print(f"Counts: {counts}")
        # Uncomment the following line to debug
        # breakpoint()
        break



