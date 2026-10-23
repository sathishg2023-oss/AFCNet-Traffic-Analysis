"""Training and evaluation loops."""
from pathlib import Path
import torch
from .metrics import regression_metrics

def run_epoch(model, loader, device, optimizer=None, amp=False):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=bool(amp and device.type == 'cuda'))
    targets, predictions, losses = [], [], []
    for images, counts in loader:
        images, counts = images.to(device), counts.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == 'cuda')):
                pred = model(images)
                loss = criterion(pred, counts)
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        losses.append(float(loss.detach().cpu()))
        targets.extend(counts.detach().cpu().numpy().tolist())
        predictions.extend(pred.detach().float().cpu().numpy().tolist())
    metrics = regression_metrics(targets, predictions)
    metrics['loss_mse_batch_mean'] = sum(losses) / max(1, len(losses))
    return metrics, targets, predictions

def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'epoch': epoch, 'metrics': metrics, 'config': config}, path)
