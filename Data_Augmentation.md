# Data Augmentation: Step-by-Step Overview

This document provides a high-level, step-by-step explanation of how **Pre-Disaster** and **Post-Disaster** building images are augmented for training a **Damage Assessment** model. We will refer to the names of the core functions and classes without presenting all the code in full.

---

## 1. Purpose & Key Steps

Data augmentation here serves two main purposes:

1. **Crop and Mask:** We isolate a specific rectangular region of interest within the original satellite image and **mask out** everything outside that region.
2. **Polygon Processing:** We **clip**, **shift**, and **rasterize** building footprints (polygons) so that they match the newly cropped image region.
3. **Resize and Tensor Conversion:** We scale images and masks to a standardized size (e.g., **1024×1024**) and then convert them to **PyTorch** tensors.

---

## 2. Dataset Structures

We have two main dataset classes, each handling a slightly different augmentation approach:

1. **`BuildingDatasetPrePostAug`**:  
   - Reads **pre-disaster** and **post-disaster** images.
   - Crops each image to a specific rectangle region.
   - Shifts the cropped image onto a **blank 1024×1024** canvas.
   - Processes corresponding polygons (for both pre- and post-disaster data).
   - Generates bounding boxes and masks.
   - Resizes the final images and targets.

2. **`BuildingDatasetRectAug`**:  
   - Reads only **post-disaster** images.
   - Masks out all pixels beyond the specified rectangle.
   - Clips polygons to that rectangle, generating bounding boxes and masks.
   - Resizes images and targets to the specified size.

---

## 3. Core Helper Functions

Several helper functions are used internally by both dataset classes. They handle the lower-level logic:

1. **`mask_pixels_outside_rectangle(image, top, left, bottom, right)`**  
   - **Masks (zeros out)** pixels outside of the `[top:bottom, left:right]` rectangle.  
   - Ensures only the region of interest remains.

2. **`clip_polygon_to_rectangle(polygon, top, left, bottom, right)`**  
   - **Clips** the given polygon to the bounding rectangle.  
   - If a polygon is partially outside the rectangle, it is intersected so that only the inside portion is retained.

3. **`polygon_to_mask(polygon, width, height)`**  
   - **Rasterizes** a polygon (or multipolygon) into a **binary mask** (`0 = background, 1 = polygon`).  
   - Particularly useful for **Mask R-CNN** training.

---

## 4. Cropping & Masking Process

### 4.1 Cropping to a Rectangle

Both **pre** and **post** images are typically **larger** than the region we want to train on. Hence, we:

1. **Read** the image using `rasterio.open(...).read()`.
2. Use **`mask_pixels_outside_rectangle`** to **retain** only the region `[top, bottom, left, right]` and **zero** everything else.

### 4.2 Preparing a 1024×1024 Canvas (for PrePost Aug)

Inside `BuildingDatasetPrePostAug`, after masking and cropping, we:

1. Convert the cropped image (now smaller) to a **PIL Image**.
2. Create a new **blank 1024×1024** canvas.
3. **Paste** the cropped image onto this canvas at the **top-left** corner (so that the region `[top:bottom, left:right]` is mapped into `[0: (bottom-top), 0: (right-left)]` on the new canvas).

This step effectively **shifts** the building region so it starts at `(0,0)` in the new image.

---

## 5. Polygon Adjustments

### 5.1 Polygon Clipping

For each polygon:

1. We call **`clip_polygon_to_rectangle(polygon, top, left, bottom, right)`**  
   - This ensures polygons fully or partially outside the rectangle are properly **clipped** or **discarded** if entirely outside.

2. If the polygon is still valid after clipping, we **shift** it according to how the image was shifted.  
   - For instance, if the image’s top-left corner is moved to `(0,0)`, we subtract the original rectangle's `(left, top)` from the polygon coordinates.

### 5.2 Handling `MultiPolygon`s

Many building footprints might be stored as **MultiPolygon**. Therefore:

- If the result of clipping yields multiple polygons (a `MultiPolygon`), each **sub-polygon** is processed individually to:
  - Extract bounding boxes.
  - Create a mask.

### 5.3 Bounding Boxes & Masks

For each valid, clipped polygon:

1. **Extract bounding box** → `[xmin, ymin, xmax, ymax]`.
2. **Rasterize** the polygon with **`polygon_to_mask`** to create a **binary mask**.

If any bounding box ends up with zero area (e.g., `xmin == xmax` or `ymin == ymax`), it is filtered out.

---

## 6. Resizing & Tensor Conversion

After we have the **cropped** / **masked** / **shifted** image and the corresponding polygons:

1. We apply a custom **`Resize`** transform, which resizes:
   - The **image** from its current size (e.g., `452×956` or similar) to `(1024×1024)`.
   - The **bounding boxes** (by scaling their coordinates).
   - The **masks** (via **nearest-neighbor** interpolation).

2. Finally, we call **`to_tensor`** (from **`torchvision.transforms.functional`**) to **convert** the resized PIL Image (and the mask arrays) into **PyTorch** tensors suitable for model training.

---

## 7. Integrating with PyTorch `Dataset` & `DataLoader`

### 7.1 `BuildingDatasetPrePostAug`

- **`__getitem__`**:
  1. Reads **both** the pre- and post-disaster images.
  2. Applies masking/cropping.
  3. Places each cropped image onto a 1024×1024 canvas.
  4. Clips, shifts, and rasterizes polygons.
  5. Resizes everything and returns a **dictionary** containing:
     - `"pre_img"`, `"pre_target"`
     - `"post_img"`, `"post_target"`

### 7.2 `BuildingDatasetRectAug`

- **`__getitem__`**:
  1. Reads the post-disaster image.
  2. Masks out everything outside `[TOP, LEFT, BOTTOM, RIGHT]`.
  3. Clips polygons accordingly.
  4. Resizes and returns:
     - `"img"`, `"target"`

### 7.3 Combining Datasets

To alternate **PrePost** and **Single**-image samples in one training loop, a **`CombinedDataset`** can be used, along with a **custom collate function** (`collate_fn_combined`) to handle the variable number of images per item.

---

## 8. Summary

- **`mask_pixels_outside_rectangle`**: Zeroes out pixels beyond the region of interest.
- **`clip_polygon_to_rectangle`**: Clips polygons to the same region.
- **`polygon_to_mask`**: Converts polygons to binary masks.
- **`BuildingDatasetPrePostAug`**: Handles **pre** + **post** images, shifting them into a uniform 1024×1024 space.
- **`BuildingDatasetRectAug`**: Simplifies the process to a single post-disaster rectangle-based approach.
- **`Resize`** transform: Maintains correct aspect ratios for bounding boxes & masks.
- **Combined** approach: Optionally merges these two datasets for more diversified training data.

This carefully designed pipeline ensures that **images, bounding boxes, and masks** all align correctly after cropping, shifting, and resizing. It also handles **edge cases** like partial polygons and multi-polygons, making it more robust for **satellite-based disaster assessment** tasks.
