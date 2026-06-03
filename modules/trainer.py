import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm


def _to_device(batch, device):
    return batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)


def train_epoch(model, data_loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    predictions, labels, probabilities = [], [], []
    use_amp = scaler is not None and scaler.is_enabled()

    for batch in tqdm(data_loader, desc="Training", leave=False):
        images, target = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, target)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        predictions.append(probs.argmax(dim=1).cpu())
        labels.append(target.detach().cpu())
        probabilities.append(probs.cpu())

    return total_loss / max(total_samples, 1), calculate_metrics(labels, predictions, probabilities)


@torch.no_grad()
def validate_model(model, data_loader, criterion, device, stage="Validation"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions, labels, probabilities = [], [], []

    for batch in tqdm(data_loader, desc=stage, leave=False):
        images, target = _to_device(batch, device)
        logits = model(images)
        loss = criterion(logits, target)
        probs = torch.softmax(logits, dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        predictions.append(probs.argmax(dim=1).cpu())
        labels.append(target.cpu())
        probabilities.append(probs.cpu())

    return total_loss / max(total_samples, 1), calculate_metrics(labels, predictions, probabilities)


def calculate_metrics(label_chunks, pred_chunks, prob_chunks):
    y_true = torch.cat(label_chunks).numpy()
    y_pred = torch.cat(pred_chunks).numpy()
    y_prob = torch.cat(prob_chunks).numpy().astype(np.float64)
    num_classes = y_prob.shape[1]
    row_sum = y_prob.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    y_prob = y_prob / row_sum
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }
    try:
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc"] = roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics
