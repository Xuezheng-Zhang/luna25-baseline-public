"""
Script for training a ResNet18 or I3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet18
from models.model_3d import I3D
from dataloader import get_data_loader
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas
from experiment_config import config
from datetime import datetime
import argparse

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights


def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,

):
    """
    Train a ResNet18 or an I3D model
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)

    print()

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )
    print()
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader_2d = get_data_loader(
        config.DATADIR,
        train_df,
        mode="2D",
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )
    
    train_loader_3d = get_data_loader(
        config.DATADIR,
        train_df,
        mode="3D",
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader_2d = get_data_loader(
        config.DATADIR,
        valid_df,
        mode="2D",
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )
    
    valid_loader_3d = get_data_loader(
        config.DATADIR,
        valid_df,
        mode="3D",
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    device = torch.device("cuda:0")

    model_2d = ResNet18().to(device)
    model_3d = I3D(
            num_classes=1,
            input_channels=3,
            pre_trained=True,
            freeze_bn=True,
        ).to(device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer_2d = torch.optim.Adam(
        model_2d.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    optimizer_3d = torch.optim.Adam(
        model_3d.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):

        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train

        model_2d.train(); model_3d.train()

        epoch_loss = 0
        step = 0

        train_loader = zip(train_loader_2d, train_loader_3d)
        num_batches = min(len(train_loader_2d), len(train_loader_3d))
                
        for batch_2d, batch_3d in tqdm(train_loader, total=num_batches):
            step += 1
            inputs, labels = batch_2d["image"], batch_2d["label"]
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            optimizer_2d.zero_grad()
            outputs_2d = model_2d(inputs)
            loss_2d = loss_function(outputs_2d.squeeze(), labels.squeeze())
            loss_2d.backward()
            optimizer_2d.step()
            
            inputs, labels = batch_3d["image"], batch_3d["label"]
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            optimizer_3d.zero_grad()
            outputs_3d = model_3d(inputs)
            loss_3d = loss_function(outputs_3d.squeeze(), labels.squeeze())
            loss_3d.backward()
            optimizer_3d.step()
            
            loss_combined = (loss_2d + loss_3d) / 2
            epoch_loss += loss_combined.item()
            epoch_len = num_batches
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.4f}".format(step, epoch_len, loss_combined.item())
                )
        epoch_loss /= step
        logging.info(
            "epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss)
        )

        # validate

        model_2d.eval(); model_3d.eval()

        epoch_loss = 0
        step = 0
        
        valid_loader = zip(valid_loader_2d, valid_loader_3d)
        num_batches = min(len(valid_loader_2d), len(valid_loader_3d))

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            for val_data_2d, val_data_3d in valid_loader:
                step += 1
                
                val_images_2d = val_data_2d["image"].to(device)
                val_labels = val_data_2d["label"].float().to(device)
                outputs_2d = model_2d(val_images_2d).squeeze()
                
                val_images_3d = val_data_3d["image"].to(device)
                outputs_3d = model_3d(val_images_3d).squeeze()
                
                ensemble_logits = (outputs_2d + outputs_3d) / 2
                
                loss = loss_function(ensemble_logits, val_labels.squeeze())
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, ensemble_logits], dim=0)
                y = torch.cat([y, val_labels], dim=0)

                epoch_len = num_batches

            epoch_loss /= step
            logging.info(
                "epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss)
            )

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)

            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)

            if auc_metric > best_metric:

                counter = 0
                best_metric = auc_metric
                best_metric_epoch = epoch + 1

                torch.save(model_2d.state_dict(), exp_save_root / "best_model_2d.pth")
                torch.save(model_3d.state_dict(), exp_save_root / "best_model_3d.pth")

                metadata = {
                    "train_csv": train_csv_path,
                    "valid_csv": valid_csv_path,
                    "config": config,
                    "best_auc": best_metric,
                    "epoch": best_metric_epoch,
                }
                np.save(
                    exp_save_root / "config.npy",
                    metadata,
                )

                logging.info("saved new best metric model")

            logging.info(
                "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                    epoch + 1, auc_metric, best_metric, best_metric_epoch
                )
            )
        counter += 1

    logging.info(
        "train completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )


if __name__ == "__main__":


    experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"

    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)

    # start training run
    train(
        train_csv_path=config.CSV_DIR_TRAIN,
        valid_csv_path=config.CSV_DIR_VALID,
        exp_save_root=exp_save_root,
        )