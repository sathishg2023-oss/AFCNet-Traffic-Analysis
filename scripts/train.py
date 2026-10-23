from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from afcnet.data import VehicleCountDataset
from afcnet.model import AFCNet
from afcnet.engine import run_epoch, save_checkpoint

def main():
    p = argparse.ArgumentParser(description='Train AFCNet on paired UA-DETRAC frames')
    p.add_argument('--images', required=True); p.add_argument('--labels', required=True)
    p.add_argument('--output', default='outputs/afcnet'); p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--amp', action='store_true'); p.add_argument('--split', choices=['contiguous','random'], default='contiguous', help='contiguous is safer for ordered video frames; keep video groups separate if available')
    a = p.parse_args()
    if not 0 < a.val_fraction < 1: p.error('--val-fraction must be between 0 and 1')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Build separate transform datasets but retain identical, sorted file pairing.
    train_ds = VehicleCountDataset(a.images, a.labels, train=True)
    val_ds = VehicleCountDataset(a.images, a.labels, train=False)
    if [x[:2] for x in train_ds.pairs] != [x[:2] for x in val_ds.pairs]: raise RuntimeError('Train/validation pairing mismatch')
    n = len(train_ds); n_val = max(1, int(round(n*a.val_fraction))); n_train = n - n_val
    if n_train < 1: p.error('Dataset too small for this validation fraction')
    if a.split == 'random':
        g = torch.Generator().manual_seed(a.seed); indices = torch.randperm(n, generator=g).tolist()
        train_idx, val_idx = indices[:n_train], indices[n_train:]
    else:
        train_idx, val_idx = list(range(n_train)), list(range(n_train, n))
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=device.type=='cuda')
    model = AFCNet(pretrained=not a.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(a).copy(); config['device'] = str(device); config['n_pairs'] = n; config['n_train'] = n_train; config['n_val'] = n_val
    (out/'config.json').write_text(json.dumps(config, indent=2))
    best = float('inf'); history=[]
    for epoch in range(1, a.epochs+1):
        tr, _, _ = run_epoch(model, train_loader, device, optimizer=optimizer, amp=a.amp)
        va, _, _ = run_epoch(model, val_loader, device, optimizer=None, amp=a.amp)
        row={'epoch':epoch,'train':tr,'validation':va}; history.append(row)
        print(f"Epoch {epoch:03d}/{a.epochs} | train MAE {tr['mae']:.4f} RMSE {tr['rmse']:.4f} | val MAE {va['mae']:.4f} RMSE {va['rmse']:.4f}")
        save_checkpoint(out/'last.pt', model, optimizer, epoch, va, config)
        if va['mae'] < best:
            best=va['mae']; save_checkpoint(out/'best.pt', model, optimizer, epoch, va, config)
        (out/'history.json').write_text(json.dumps(history, indent=2))
    print(f'Best validation MAE: {best:.4f}; checkpoint: {out / "best.pt"}')
if __name__ == '__main__': main()
