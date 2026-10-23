from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from afcnet.data import VehicleCountDataset
from afcnet.model import AFCNet
from afcnet.engine import run_epoch

def main():
    p=argparse.ArgumentParser(description='Evaluate a saved AFCNet checkpoint')
    p.add_argument('--images',required=True); p.add_argument('--labels',required=True)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',default='outputs/evaluation')
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=VehicleCountDataset(a.images,a.labels,train=False)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=AFCNet(pretrained=False).to(device); model.load_state_dict(ckpt.get('model_state_dict',ckpt)); model.eval()
    metrics, targets, preds=run_epoch(model,loader,device)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['image','ground_truth_count','predicted_count'])
        for (img,_,_),yhat,y in zip(ds.pairs,preds,targets): w.writerow([str(img),y, yhat])
    (out/'metrics.json').write_text(__import__('json').dumps(metrics,indent=2))
    print(metrics); print(f'Wrote {out / "predictions.csv"} and {out / "metrics.json"}')
if __name__=='__main__': main()
