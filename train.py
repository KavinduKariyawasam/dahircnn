import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
import gc
from dataset import BuildingDataset, collate_fn, TestDataset, test_collate_fn, train_df_create, test_df_create
from model import CustomMaskRCNNBackbone, CUSTOM_MaskRCNN, Custom_Transformer_UNet
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import coloredlogs

LOG_LEVEL_STYLES = {
    'debug': {'color': 'blue'},
    'info': {'color': 'green'},  
    'warning': {'color': 'yellow'},
    'error': {'color': 'red'},
    'critical': {'color': 'red', 'bold': True}
}

coloredlogs.install(
    level=logging.INFO,  # Set the minimum log level
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    level_styles=LOG_LEVEL_STYLES
)

EPOCHS = 10
BATCH_SIZE = 2
DEVICE = ('cuda:1' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
DATASET_PATH = "/home/deependra/Dataset"

def load_model(device, checkpoint_path=None):
    backbone = Custom_Transformer_UNet(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                                with_pos='learned', with_decoder_pos='learned', enc_depth=1, dec_depth=8).to(device)

    custom_backbone = CustomMaskRCNNBackbone(backbone)
    
    model = CUSTOM_MaskRCNN(
        backbone=custom_backbone,
        num_classes=5
    )

    del backbone, custom_backbone
    gc.collect()
    torch.cuda.empty_cache()
    
    model.transform.min_size = (1024,)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Model weights loaded from {checkpoint_path}")

    return model

def load_or_create_dataframe(csv_path, create_func, **kwargs):
    if os.path.exists(csv_path):
        logging.debug(f"Loading {csv_path}")
        return pd.read_csv(csv_path)
    
    logging.debug(f"Creating {csv_path}")
    df = create_func(**kwargs)
    df.to_csv(csv_path, index=False) 
    return df

def load_data(root_dir, batch_size=4, split='tier1'):
    dataset_paths = {
        "train": f"{root_dir}/train_df_{split}.csv",
        "val": f"{root_dir}/train_df_hold.csv",
        "test": f"{root_dir}/test_df.csv",
    }

    train_df = load_or_create_dataframe(dataset_paths["train"], train_df_create, root=root_dir)
    val_df = load_or_create_dataframe(dataset_paths["val"], train_df_create, root=root_dir)
    test_df = load_or_create_dataframe(dataset_paths["test"], test_df_create, root=root_dir)

    def create_dataloader(dataset, batch_size, shuffle, collate_fn):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn
        )

    train_loader = create_dataloader(BuildingDataset(train_df, resize_size=(512, 512)), batch_size, True, collate_fn)
    val_loader = create_dataloader(BuildingDataset(val_df, resize_size=(1024, 1024)), 1, False, collate_fn)
    test_loader = create_dataloader(TestDataset(test_df, resize_size=(1024, 1024)), 1, False, test_collate_fn)

    logging.info("Data loading complete")
    return train_loader, test_loader, val_loader

def building_count_mae_loss(model, images, gt, score_threshold=0.5):
    model.eval()
    
    with torch.no_grad():
        outputs = model(images)
        
        for i, output in enumerate(outputs):
            scores = output["scores"].cpu()
            keep = scores >= score_threshold
            labels = output["labels"][keep].cpu().numpy()

            counts = {class_name: 0 for class_name in damage_class_to_id.keys()}
            
            for label in labels:
                class_name = id_to_damage_class.get(label, "unknown")
                if class_name in counts:
                    counts[class_name] += 1
            
            mae = 0
        for class_name in counts.keys():
            mae += abs((counts[class_name]) - gt[0].get(class_name, 0))
        
        total_count = sum(gt[0].values())
        if total_count == 0:
            total_count = 1
        mae /= total_count
    return mae
    

def building_count_mae_loss_differentiable(model, images, gt, score_threshold=0.5):
    model.eval()
    mae_total = torch.tensor(0.0, device=images[0][0].device, requires_grad=True)  
    batch_size = len(images)

    outputs = model(images)

    for i, output in enumerate(outputs):
        scores = output["scores"]  
        keep = scores >= score_threshold  
        predicted_labels = output["labels"][keep] 
        unique_labels, counts = torch.unique(predicted_labels, return_counts=True)

        predicted_counts = torch.zeros(len(damage_class_to_id), device=images[0][0].device)
    
        for j, label in enumerate(unique_labels):
            class_idx = label.item()  
            if class_idx < len(predicted_counts):  
                predicted_counts[class_idx] = counts[j] 

        gt_counts = torch.zeros(len(damage_class_to_id), device=images[0][0].device)
        for class_name, count in gt[i].items():
            class_idx = damage_class_to_id[class_name]
            gt_counts[class_idx - 1] = count

        mae = torch.abs(predicted_counts - gt_counts).sum()

        total_count = gt_counts.sum()
        total_count = torch.where(total_count > 0, total_count, torch.tensor(1.0, device=images[0][0].device))  
        mae /= total_count

        mae_total = mae_total + mae  
    return mae_total / batch_size if batch_size > 0 else torch.tensor(0.0, device=images[0][0].device, requires_grad=True)


def train_model(model, train_loader, val_loader=None, device=DEVICE, epochs=10, plot_loss=True):
    logging.info("Starting training")
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()
    
    loss_list = []
    failed_count = 0
    
    best_val_loss = 10000000
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for i, (images, targets, counts) in epoch_progress:
            try:
                with autocast():
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    
                    weight_classifier = 2.0  
                    loss_dict['loss_classifier'] *= weight_classifier
                    
                    losses = sum(loss for loss in loss_dict.values())
                    
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                epoch_loss += losses.item()
                torch.cuda.empty_cache()

                epoch_progress.set_postfix({"Iteration Loss": losses.item(), "Epoch Loss": epoch_loss / (i + 1)})

            except Exception as e:
                logging.warning(f"Failed iteration: {i + 1}, {e}")
                failed_count += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_list.append(avg_epoch_loss)
        logging.info(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_epoch_loss}")

            
        if val_loader is not None:
            logging.info("Validating model")
            model.eval()
            image_counts = []
            
            validate_progress = tqdm(enumerate(val_loader), total=len(val_loader), unit="batch")
            
            mae_total_loss = 0
            
            with torch.no_grad():
                for i, (images, targets, target_counts) in validate_progress:
                    imgs = list(image.to(device) for image in images)

                    outputs = model(imgs)

                    for i, output in enumerate(outputs):
                        img_ids = targets[0]['image_id']
                        img_id = img_ids[i]

                        scores = output["scores"].cpu()
                        keep = scores >= 0.5
                        labels = output["labels"][keep].cpu().numpy()

                        counts = {class_name: 0 for class_name in damage_class_to_id.keys()}
                        for label in labels:
                            class_name = id_to_damage_class.get(label, "unknown")
                            if class_name in counts:
                                counts[class_name] += 1

                        image_counts = [{"img_id": img_id, "counts": counts}]
                        
                    total_categories = len(target_counts) 
                    absolute_errors = [
                        abs(target_counts[0][key] - image_counts[0]['counts'][key]) for key in target_counts[0]
                    ]
                    mae_loss = sum(absolute_errors) / total_categories
                    mae_total_loss += mae_loss

                    validate_progress.set_postfix({"Iteration Loss": mae_loss})
                    
            mae_total_loss /= len(val_loader)
            logging.info(f"Validation MAE Loss: {mae_total_loss}")
            
            if mae_total_loss < best_val_loss:
                best_val_loss = mae_total_loss
                best_epoch = epoch
                best_model_path = CHECKPOINT_PATH
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model saved at: {best_model_path}")
            
            model.train()
    
    model_path = f"./checkpoints/last_model.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Last model saved at: {model_path}")
            
    logging.info("Training sucessfully completed")
    if len(loss_list) > 0:
        logging.info(f"Final training loss: {loss_list[-1]}")
    logging.debug(f"Failed iterations: {failed_count/epochs}")
    logging.info(f"Best model found at epoch: {best_epoch}")
        
    if plot_loss:
        plt.plot(loss_list)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")

        plt.savefig("./plots/training_loss.png")
        plt.close()
    
    model = load_model(device=device, checkpoint_path=best_model_path)

    return model


damage_class_to_id = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
id_to_damage_class = {v: k for k, v in damage_class_to_id.items()}


def predict_and_count(model, data_loader, device, score_threshold=0.3):
    model.eval()
    image_counts = []

    with torch.no_grad():
        for imgs, img_ids in tqdm(data_loader, desc="Processing images"):
            imgs = list(image.to(device) for image in imgs)

            outputs = model(imgs)

            for i, output in enumerate(outputs):
                img_id = img_ids[i]

                scores = output["scores"].cpu()
                keep = scores >= score_threshold
                labels = output["labels"][keep].cpu().numpy()

                counts = {class_name: 0 for class_name in damage_class_to_id.keys()}
                for label in labels:
                    class_name = id_to_damage_class.get(label, "unknown")
                    if class_name in counts:
                        counts[class_name] += 1

                image_counts.append({"img_id": img_id, "counts": counts})

    return image_counts

def create_submission(predictions, save_path):
    rows = []

    for entry in predictions:
        img_id = entry["img_id"]
        counts = entry["counts"]
        for damage_class, count in counts.items():
            damage_class_formatted = damage_class.replace("-", "_")
            row_id = f"{img_id}_X_{damage_class_formatted}"
            rows.append({"id": row_id, "target": count})

    sub_df = pd.DataFrame(rows)
    sub_df = sub_df.sort_values("id").reset_index(drop=True)
    
    if not os.path.exists("./submissions"):
        os.makedirs("./submissions")
    
    sub_df.to_csv(save_path, index=False)
    logging.info(f"Submission file saved at {save_path}")
    return sub_df

if __name__ == "__main__":
    device = DEVICE
    train_loader, test_loader, val_loader = load_data(root_dir=DATASET_PATH, batch_size=BATCH_SIZE, split="tier1")
    
    resume_training = False
    training = True
    
    if training:
        model = load_model(device=device)
        if resume_training:
            checkpoint_path = CHECKPOINT_PATH
            model = load_model(device=device, checkpoint_path=checkpoint_path)

        model = train_model(model, train_loader, val_loader=val_loader, device=DEVICE, epochs=EPOCHS)
    else:
        checkpoint_path = CHECKPOINT_PATH
        model = load_model(device=device, checkpoint_path=checkpoint_path)
    
    model = model.to(device)

    predictions = predict_and_count(model, test_loader, device, score_threshold=0.5)

    sub_df = create_submission(predictions, "./submissions/submission_tier1_512.csv")