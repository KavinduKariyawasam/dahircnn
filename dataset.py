from pathlib import Path
import pandas as pd 
import numpy  as np 
import json
from collections import Counter
from shapely.wkt import loads
from shapely.geometry import Polygon, box
# import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster
from rasterio.transform import from_bounds
from tqdm.notebook import tqdm

# Visulazation  
import  matplotlib.pyplot as plt 
from PIL import Image, ImageDraw
import os


from pathlib import Path
import pandas as pd


# PyTorch and computer vision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
# from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
# from torchvision.transforms import functional as F

from torchvision.transforms import functional as TF

MEAN_TRAIN = torch.tensor([0.3052, 0.3381, 0.2510, 0.3073, 0.3427, 0.2542])
STD_TRAIN = torch.tensor([0.1630, 0.1449, 0.1360, 0.1609, 0.1426, 0.1348])

MEAN_TEST = torch.tensor([0.1395, 0.1422, 0.1433, 0.1370, 0.1298, 0.1318])
STD_TEST = torch.tensor([0.2325, 0.2271, 0.2217, 0.2182, 0.2048, 0.2003])


DATA_DIR = "/home/deependra/Dataset"

def normalize(image_tensor, mean, std):
    return TF.normalize(image_tensor, mean, std)

# Extract and add lat and lon bounds - to be used in downloading data from aws
def parse_coords(coord_str):
    # Remove parentheses
    coord_str = coord_str.strip("()")
    # Split by the semicolon
    lat_str, lon_str = coord_str.split(";")
    # Convert to float
    lat = float(lat_str)
    lon = float(lon_str)
    return lat, lon


def test_df_create(root="/home/deependra/Dataset"):
# Load test image coords and image paths
    test_coords = pd.read_csv(f"{root}/malawi_test/test_image_coords.csv")
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


def train_df_create(root, split="hold"):
    dfs = []
    image_dir = Path(f"{root}/x2view/geotiffs/{split}/images")
    label_dir = Path(f"{root}/x2view/geotiffs/{split}/labels")
    
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
    # merged_df["dataset_type"] = dataset_path

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
    df = df.drop("un-classified", axis=1)
    df.to_csv(f"/home/deependra/Dataset/malawi_test/train_df_{split}.csv", index=False)
    return df

def get_subtype_counts(label_path):
    with open(label_path, "r") as file:
        label_data = json.load(file)

    subtypes = [
        building["properties"]["subtype"] for building in label_data["features"]["xy"]
    ]
    return Counter(subtypes)


class Resize(object):
    def __init__(self, size):
        self.size = size  # size should be a tuple (height, width)

    def __call__(self, image, target):
        orig_width, orig_height = image.size
        image = TF.resize(image, self.size)

        ratio_width = self.size[0] / orig_width
        ratio_height = self.size[1] / orig_height

        if "boxes" in target:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_height
            target["boxes"] = boxes

        if "masks" in target:
            masks = target["masks"]
            masks = masks.unsqueeze(1)  # Add channel dimension
            masks = TF.resize(
                masks, self.size, interpolation=TF.InterpolationMode.NEAREST
            )
            masks = masks.squeeze(1)
            target["masks"] = masks

        return image, target


class BuildingDataset(Dataset):
    def __init__(self, df, resize_size=(512, 512)):
        self.df = df.reset_index(drop=True)
        self.resize_size = resize_size
        self.augment = False
        self.damage_class_to_id = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
        }
        # self.augmentations = TF.Compose([
        #     TF.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.post_image_path
        img_pre_path = row.pre_image_path
        label_path = row.post_label_path
        image_id = torch.tensor([idx])
        damge_counts = {
                "no-damage": row["no-damage"],
                "minor-damage": row["minor-damage"],
                "major-damage": row["major-damage"],
                "destroyed": row["destroyed"],
            }
        
        try:
            with rasterio.open(img_pre_path) as src:
                img_array = src.read()
                if img_array.shape[0] >= 3:
                    img_array = img_array[:3, :, :]
                else:
                    img_array = np.repeat(img_array, 3, axis=0)

                img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
                img_pre = Image.fromarray(img_array)
            
            with rasterio.open(img_path) as src:
                img_array = src.read()
                if img_array.shape[0] >= 3:
                    img_array = img_array[:3, :, :]
                else:
                    img_array = np.repeat(img_array, 3, axis=0)

                img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
                img = Image.fromarray(img_array)
        except Exception as e:
            print(f"Error reading image: {img_path}")
            print(e)
            return {
                "img": None,
                "target": {'image_id': image_id},
                "counts": None
            }

        width, height = img.size

        with open(label_path, "r") as f:
            annotations = json.load(f)

        boxes = []
        labels = []
        masks = []
        annotations = annotations["features"]["xy"]

        for annotation in annotations:
            properties = annotation["properties"]
            subtype = properties["subtype"]
            damage_label = self.damage_class_to_id.get(subtype, 0)

            polygon_wkt = annotation["wkt"]
            polygon = loads(polygon_wkt)

            if not polygon.is_valid:
                continue

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

        resize_transform = Resize(self.resize_size)
        img_pre, _ = resize_transform(img_pre, {})
        img, target = resize_transform(img, target)

        # Use TF.to_tensor instead of F.to_tensor
        img_pre = TF.to_tensor(img_pre)
        img = TF.to_tensor(img)
        
        # img_pre = normalize(img_pre, MEAN_TRAIN[3:], STD_TRAIN[3:])
        # img = normalize(img, MEAN_TRAIN[:3], STD_TRAIN[:3])
        
        if self.augment:
            img = self.augmentations(img)
            img_pre = self.augmentations(img_pre)
        
        imgs = torch.cat([img, img_pre], dim=0)

        return {
            "img": imgs,
            "target": target,
            "counts": damge_counts
        }

    def __len__(self):
        return len(self.df)

    def _polygon_to_mask(self, polygon, width, height):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        if polygon.is_empty:
            return np.array(mask, dtype=np.uint8)

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

        return np.array(mask, dtype=np.uint8)

    def _clip_polygon_to_image(self, polygon, width, height):
        image_box = box(0, 0, width, height)
        return polygon.intersection(image_box)
    

# Test dataset class
class TestDataset(Dataset):
    def __init__(self, df, resize_size=(512, 512)):
        self.df = df.reset_index(drop=True)
        self.resize_size = resize_size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.post_image_path
        img_pre_path = row.pre_image_path
        img_id = row.id

        with rasterio.open(img_pre_path) as src:
            img_array = src.read()
            if img_array.shape[0] >= 3:
                img_array = img_array[:3, :, :]
            else:
                img_array = np.repeat(img_array, 3, axis=0)

            img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
            img_pre = Image.fromarray(img_array)
        
        # Load image with rasterio
        with rasterio.open(img_path) as src:
            img_array = src.read()  # shape: (C, H, W)

            # Ensure at least 3 channels (R,G,B)
            if img_array.shape[0] >= 3:
                img_array = img_array[:3, :, :]
            else:
                # If fewer than 3 channels, replicate to make it appear like an RGB image
                img_array = np.repeat(img_array, 3, axis=0)

            # Convert to H x W x C and uint8
            img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img_array)

        # Resize and convert to tensor
        img = TF.resize(img, self.resize_size)
        img = TF.to_tensor(img)
        
        img_pre = TF.resize(img_pre, self.resize_size)
        img_pre = TF.to_tensor(img_pre)

        imgs = torch.cat([img, img_pre], dim=0)
        
        return {
            "img": imgs,
            "img_id": img_id
        }

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    # Extract each component from the batch of dictionaries
    imgs = [item["img"] for item in batch]
    targets = [item["target"] for item in batch]
    counts = [item["counts"] for item in batch]
    return imgs, targets, counts

def test_collate_fn(batch):
    imgs = [item["img"] for item in batch]
    img_ids = [item["img_id"] for item in batch]

    return imgs,img_ids

