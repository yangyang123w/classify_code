import argparse
import json
import logging
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    module=r"pydantic\._internal\._generate_schema",
    message=r".*was provided to the `Field\(\)` function.*",
)

import torch
import torch.nn as nn

from dataloader.loader import get_dataloaders
from models.model import BackboneClassifier
from modules.trainer import train_epoch, validate_model
from utils import (
    append_metrics_csv,
    count_parameters,
    create_exp_dir,
    plot_confusion_matrix,
    plot_training_trend,
    save_args,
    set_random_seed,
    setup_logger,
)


def get_arguments():
    parser = argparse.ArgumentParser("Backbone Classification Training")
    parser.add_argument("--csv_path", type=str, default="/sdb1/liran/downsteam_code/classify/splits/M1_split.csv")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--backbone", type=str, default="openus", choices=["usfm", "fetalclip", "dinov3", "openus"])
    parser.add_argument("--usfm_weight", type=str, default="/sdb1/liran/downsteam_code/classify/classify_backbone_project/USFM_latest.pth")
    parser.add_argument("--fetalclip_weight", type=str, default="/sdb1/liran/compare_code/fetalclip/FetalCLIP_weights.pt")
    parser.add_argument("--dinov3_weight", type=str, default="/sdb1/liran/downsteam_code/teacher_checkpoint2.pth")
    parser.add_argument("--openus_weight", type=str, default="/sdb1/liran/downsteam_code/classify/classify_backbone_project/openus_cpt0150.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="runs")
    return parser.parse_args()


def setup_experiment(args):
    set_random_seed(args.seed)
    exp_name = f"{args.backbone}-EXP-{time.strftime('%Y%m%d-%H%M%S')}"
    exp_dir = Path(args.save_dir) / exp_name
    ckp_dir = exp_dir / "checkpoint"
    create_exp_dir(ckp_dir)
    setup_logger(exp_dir)
    save_args(args, exp_dir)
    logging.info("Experiment arguments: %s", args)
    return exp_dir, ckp_dir


def initialize(args, device):
    train_loader, val_loader, test_loader, class_names = get_dataloaders(args)
    args.class_names = class_names
    args.num_classes = len(class_names)
    logging.info("class_names: %s", class_names)
    logging.info(
        "dataset sizes | train=%d val=%d test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = BackboneClassifier(args, num_classes=args.num_classes).to(device)
    total_params, trainable_params = count_parameters(model)
    logging.info("model parameters | total=%.2fM trainable=%.2fM", total_params / 1e6, trainable_params / 1e6)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    return model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, scaler


def main():
    args = get_arguments()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    exp_dir, ckp_dir = setup_experiment(args)
    model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, scaler = initialize(args, device)

    best_val_f1 = -1.0
    no_improvement_epochs = 0
    train_metrics = {name: [] for name in ["loss", "accuracy", "f1", "precision", "recall", "auc"]}
    val_metrics = {name: [] for name in ["loss", "accuracy", "f1", "precision", "recall", "auc"]}

    for epoch in range(1, args.epochs + 1):
        logging.info("-" * 100)
        logging.info("Epoch: %d/%d | LR: %.6g", epoch, args.epochs, optimizer.param_groups[0]["lr"])
        train_loss, train_epoch_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_epoch_metrics = validate_model(model, val_loader, criterion, device, stage="Validation")
        test_loss, test_epoch_metrics = validate_model(model, test_loader, criterion, device, stage="Testing")
        scheduler.step()

        logging.info("Train | Loss: %.4f | Metrics: %s", train_loss, train_epoch_metrics)
        logging.info("Val   | Loss: %.4f | Metrics: %s", val_loss, val_epoch_metrics)
        logging.info("Test  | Loss: %.4f | Metrics: %s", test_loss, test_epoch_metrics)

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for prefix, loss, metrics in [
            ("train", train_loss, train_epoch_metrics),
            ("val", val_loss, val_epoch_metrics),
            ("test", test_loss, test_epoch_metrics),
        ]:
            row[f"{prefix}_loss"] = loss
            for key in ["accuracy", "precision", "recall", "f1", "auc"]:
                row[f"{prefix}_{key}"] = metrics.get(key, float("nan"))
        append_metrics_csv(exp_dir / "metrics.csv", row)

        train_metrics["loss"].append(train_loss)
        val_metrics["loss"].append(val_loss)
        for metric in ["accuracy", "f1", "precision", "recall", "auc"]:
            train_metrics[metric].append(train_epoch_metrics.get(metric, float("nan")))
            val_metrics[metric].append(val_epoch_metrics.get(metric, float("nan")))
        for metric_name in ["Loss", "Accuracy", "F1", "Precision", "Recall", "AUC"]:
            key = metric_name.lower()
            plot_training_trend(train_metrics[key], val_metrics[key], metric_name, exp_dir)

        torch.save({"epoch": epoch, "model": model.state_dict(), "class_names": args.class_names}, ckp_dir / "last_model.pth")
        current_val_f1 = val_epoch_metrics.get("f1", 0.0)
        if current_val_f1 <= best_val_f1:
            no_improvement_epochs += 1
            logging.info(
                "No improvement | Current Val F1: %.4f | Best: %.4f | Patience: %d/%d",
                current_val_f1,
                best_val_f1,
                no_improvement_epochs,
                args.patience,
            )
            if no_improvement_epochs >= args.patience:
                logging.info("Early stopping triggered.")
                break
            continue

        no_improvement_epochs = 0
        best_val_f1 = current_val_f1
        torch.save({"epoch": epoch, "model": model.state_dict(), "class_names": args.class_names}, ckp_dir / "best_model.pth")
        best_test_metrics = dict(test_epoch_metrics)
        best_test_metrics["epoch"] = epoch
        with open(exp_dir / "best_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_test_metrics, f, ensure_ascii=False, indent=2)
        plot_confusion_matrix(
            test_epoch_metrics["confusion_matrix"],
            exp_dir / "best_confusion_matrix.png",
            class_names=args.class_names,
            normalize=False,
            title=f"Best Confusion Matrix (Epoch {epoch})",
        )
        plot_confusion_matrix(
            test_epoch_metrics["confusion_matrix"],
            exp_dir / "best_confusion_matrix_normalized.png",
            class_names=args.class_names,
            normalize=True,
            title=f"Best Confusion Matrix Normalized (Epoch {epoch})",
        )
        logging.info("Saved best model | Val F1: %.4f | Test F1: %.4f", best_val_f1, test_epoch_metrics.get("f1", 0.0))


if __name__ == "__main__":
    main()
