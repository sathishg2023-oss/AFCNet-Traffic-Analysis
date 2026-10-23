import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from afcnet.model import AFCNet

def test_forward_shape():
    model = AFCNet(pretrained=False).eval()
    with torch.no_grad(): y = model(torch.randn(2,3,224,224))
    assert tuple(y.shape) == (2,)
