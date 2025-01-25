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

def load_model(checkpoint_path=None):
    device = ('cuda:1' if torch.cuda.is_available() else 'cpu')

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
        print(f"Model weights loaded from {checkpoint_path}")


    return model

def load_data():
    if os.path.exists("/home/deependra/Dataset/malawi_test/test_df.csv"):
        print("Loading test_df from file")
        test_df = pd.read_csv("/home/deependra/Dataset/malawi_test/test_df.csv")
    else:
        print("Creating test_df")
        test_df = test_df_create()

    if os.path.exists("/home/deependra/Dataset/malawi_test/train_df.csv"):
        print("Loading train_df from file")
        train_df = pd.read_csv("/home/deependra/Dataset/malawi_test/train_df.csv")
    else:
        print("Creating train_df")
        train_df = train_df_create()

    # Instantiate dataset
    train_dataset = BuildingDataset(train_df, resize_size=(1024, 1024))

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    print("Test data loading complete")
    
    test_dataset = TestDataset(test_df, resize_size=(1024, 1024))
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=test_collate_fn,
    )

    print("Train data loading complete")

    return train_loader, test_loader

def building_count_mae_loss(model, images, gt, score_threshold=0.5):
    model.eval()
    # device = ('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    
    # print(f"MAE loss calculation started")
    
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
        # mae /= 4
        # print(f"gt: {gt[0]}, pred: {counts}, MAE: {mae}")
    return mae
    

def train_model(model, train_loader, val_loader=None, epochs=10):
    device = ('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()
    
    loss_list = []
    loss_avg_dict = {}
    failed_count = 0
    
    for epoch in range(epochs):
        # Initialize tqdm for epoch-wise progress tracking
        epoch_loss = 0
        epoch_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for i, (images, targets, counts) in epoch_progress:
            try:
                with autocast():
                    # Prepare images and targets
                    images = images[0]
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # Compute loss
                    loss_dict = model(images, targets)
                    # losses = sum(loss for loss in loss_dict.values())
                    
                    mae_loss = building_count_mae_loss(model, images, counts)
                    model.train()
                    # print(f"MAE: {mae_loss}")
                    
                    # breakpoint()
                    
                    weight_classifier = 2.0  
                    loss_dict['loss_classifier'] *= weight_classifier

                    losses = sum(loss for loss in loss_dict.values())
                    losses += mae_loss
                    
                # breakpoint()

                # Backpropagation and optimization
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update cumulative loss
                epoch_loss += losses.item()
                torch.cuda.empty_cache()

                # Update tqdm description for the current iteration
                epoch_progress.set_postfix({"Iteration Loss": losses.item(), "Epoch Loss": epoch_loss / (i + 1)})

            except Exception as e:
                # Log errors if any
                failed_count += 1
                # print(f"Error: {e}")
                continue

        # Log average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_list.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_epoch_loss}")

        if (epoch + 1) % 5 == 0:
            # Save the model
            model_path = f"/home/deependra/kuyesera/dahircnn/checkpoints_mae/model_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at: {model_path}")
            
    print("========= Training completed =============")
    if len(loss_list) > 0:
        print(f"Final training loss: {loss_list[-1]}")
    print(f"Failed iterations: {failed_count/epochs}")
    print("===========================================")
            
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")

    plt.savefig("/home/deependra/kuyesera/dahircnn/loss_plot_mae.png")
    plt.close()
    
    # for epoch in range(epochs):
    #     iteration_loss = 0
    #     for i, (images, targets) in enumerate(train_loader):
    #         try:
    #             with autocast():
    #                 # print(f"Image size: {images[0][0].shape, images[0][1].shape}")
    #                 images = images[0]
    #                 images = list(image.to(device) for image in images)
    #                 # print(f"len(images): {len(images)}")
    #                 targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #                 loss_dict = model(images, targets)
    #                 losses = sum(loss for loss in loss_dict.values())
    #             # optimizer.zero_grad()
    #             # losses.backward()
    #             # optimizer.step()
    #             scaler.scale(losses).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad()
    #             iteration_loss += losses
    #             torch.cuda.empty_cache()
    #             print(f"Iteration: {i}, Loss: {losses}")
    #         except Exception as e:
    #             print(f"Error: {e}")
    #             continue
        
    #     print(f"Epoch: {epoch + 1}/{epochs}, Loss: {iteration_loss/len(train_loader)}")
    #     torch.save(model.state_dict(), f"/home/deependra/kuyesera/dahircnn/model_{epoch}.pth")
    #     print(f"Model saved at epoch: {epoch}")
        
        # TODO: Add validation part
        # model.eval()
        # for i, (images, targets) in enumerate(test_loader):
        #     images = list(image.to(device) for image in images)
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #     loss_dict = model(images, targets)
        #     losses = sum(loss for loss in loss_dict.values())
        #     print(f"Test Iteration: {i}, Loss: {losses}")
        # model.train()
    return model


damage_class_to_id = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
id_to_damage_class = {v: k for k, v in damage_class_to_id.items()}


# Prediction function
def predict_and_count(model, data_loader, device, score_threshold=0.5):
    """
    Run inference with the model on the test data and count the number of buildings per class per image.

    Args:
        model (nn.Module): The trained Mask R-CNN model.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run inference on (CPU or GPU).
        score_threshold (float): Minimum score threshold for counting a prediction.

    Returns:
        list of dict: A list of dictionaries, each containing:
            {
              "img_id": <image identifier>,
              "counts": {
                "no-damage": int,
                "minor-damage": int,
                "major-damage": int,
                "destroyed": int
              }
            }
    """
    model.eval()
    image_counts = []

    with torch.no_grad():
        # Wrap the data_loader with tqdm for a progress bar
        for imgs, img_ids in tqdm(data_loader, desc="Processing images"):
            # Move images to the device
            img = imgs[0][0].to(device)
            img_pre = imgs[0][1].to(device)
            imgs = [img, img_pre]

            # Run the model
            outputs = model(imgs)

            for i, output in enumerate(outputs):
                img_id = img_ids[i]

                # Filter predictions by score threshold
                scores = output["scores"].cpu()
                keep = scores >= score_threshold
                labels = output["labels"][keep].cpu().numpy()

                # Count instances per class
                counts = {class_name: 0 for class_name in damage_class_to_id.keys()}
                for label in labels:
                    class_name = id_to_damage_class.get(label, "unknown")
                    if class_name in counts:
                        counts[class_name] += 1

                # Store the results for this image
                image_counts.append({"img_id": img_id, "counts": counts})

    return image_counts

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    
    dahircnn = load_model()
    # print("Model loaded: ", dahircnn)
    
    # model = train_model(dahircnn, train_loader, epochs=20)
    
    checkpoint_path = "/home/deependra/kuyesera/dahircnn/checkpoints_mae/model_5.pth"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path)
    model = model.to(device)

    # Run predictions
    print("========= Prediction started ================")
    predictions = predict_and_count(model, test_loader, device, score_threshold=0.5)
    print("============== Prediction completed =================")

    # Create submission DataFrame
    rows = []

    # Loop over each image's counts
    for entry in predictions:
        img_id = entry["img_id"]
        counts = entry["counts"]

        # For each damage class, create a row
        for damage_class, count in counts.items():
            damage_class_formatted = damage_class.replace("-", "_")
            row_id = f"{img_id}_X_{damage_class_formatted}"
            rows.append({"id": row_id, "target": count})

    # Create the DataFrame
    sub_df = pd.DataFrame(rows)

    # Sort the DataFrame by 'id'
    sub_df = sub_df.sort_values("id").reset_index(drop=True)

    # Save the DataFrame to a CSV file
    submission_path = "/home/deependra/kuyesera/dahircnn/submissions/initial_mae_5.csv"
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission file saved at {submission_path}")