import torch
from torch import nn
from torchmore import flex, layers
import torch.nn.functional as F
import os
import sys
import glob
import re

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53

def make(name, *args, device=default_device, **kw):
    model = eval("make_"+name)(*args, **kw)
    if device is not None:
        model.to(device)
    model.model_name = name
    return model

def extract_save_info(fname):
    fname = re.sub(r'.*/', '', fname)
    match = re.search(r'([0-9]{3})+-([0-9]{9})', fname)
    if match:
        return int(match.group(1)), float(match.group(2))*1e-6
    else:
        return 0, -1

def load_latest(model, pattern=None, error=False):
    if pattern is None:
        name = model.model_name
        pattern = f"models/{name}-*.pth"
    saves = sorted(glob.glob(pattern))
    if error:
        assert len(saves)>0, f"no {pattern} found"
    elif len(saves)==0:
        print(f"no {pattern} found", file=sys.stderr)
        return 0, -1
    else:
        print(f"loading {saves[-1]}", file=sys.stderr)
        model.load_state_dict(torch.load(saves[-1]))
        return extract_save_info(saves[-1])

################################################################
# ## layer combinations
# ###############################################################

def conv2d(d, r=3, stride=1, repeat=1):
    """Generate a conv layer with batchnorm and optional maxpool."""
    result = []
    for i in range(repeat):
        result += [
            flex.Conv2d(d, r, padding=(r//2, r//2), stride=stride),
            flex.BatchNorm2d(),
            nn.ReLU()
        ]
    return result

def conv2mp(d, r=3, mp=None, repeat=1):
    """Generate a conv layer with batchnorm and optional maxpool."""
    result = conv2d(d, r, repeat=repeat)
    if mp is not None:
        result += [nn.MaxPool2d(mp)]
    return result

def conv2fmp(d, r=3, fmp=(0.7, 0.85), repeat=1):
    """Generate a conv layer with batchnorm and optional fractional maxpool."""
    result = conv2d(d, r, repeat=repeat)
    if fmp is not None:
        result += [nn.FractionalMaxPool2d(3, output_ratio=fmp)]
    return result

def conv2x(d, r=3, mp=2):
    """Generate a pair of conv layers with batchnorm and maxpool."""
    result = [
        flex.Conv2d(d, r, padding=(r//2, r//2)),
        flex.BatchNorm2d(),
        nn.ReLU(),
        flex.Conv2d(d, r, padding=(r//2, r//2)),
        flex.BatchNorm2d(),
        nn.ReLU()
    ]
    if mp is not None:
        result += [nn.MaxPool2d(mp)]
    return result

def project_and_lstm(d, noutput, num_layers=1):
    return [
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(d, bidirectional=True, num_layers=num_layers),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", "BLD")
    ]

def project_and_conv1d(d, noutput, r=5):
    return [
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.Conv1d(d, r),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", "BLD")
    ]

################################################################
# ## new layer types
# ###############################################################

class UnetLayer(nn.Module):
    """Resolution pyramid layer using convolutions and upscaling.
    """
    def __init__(self, d, r=3, sub=None):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.up = flex.ConvTranspose2d(d, r, stride=2, padding=1, output_padding=1)
        if isinstance(sub, list):
            sub = nn.Sequential(*sub)
        self.sub = sub
    def forward(self, x):
        b, d, h, w = x.size()
        assert h%2==0 and w%2==0, x.size()
        lo = self.down(x)
        lo1 = self.sub(lo)
        hi = self.up(lo1)
        result = torch.cat([x, hi], dim=1)
        return result

class KeepSize(nn.Module):
    """Run layers, then upsample back to the original.
    """
    def __init__(self, mode="bilinear", sub=None, dims=None):
        super().__init__()
        if isinstance(sub, list):
            sub = nn.Sequential(*sub)
        self.sub = sub
        self.mode = mode
        self.dims = dims
    def forward(self, x):
        y = self.sub(x)
        if self.dims is None:
            size = x.size()[2:]
        else:
            size = [x.size(i) for i in self.dims]
        kw = dict(align_corners=False) if self.mode != "nearest" else {}
        try:
            return F.interpolate(y, size=size, mode=self.mode, **kw)
        except Exception as exn:
            print("error:", x.size(), y.size(), self.dims, size, self.mode, file=sys.stderr)
            raise exn

class NoopSub(nn.Module):
    def __init__(self, *args, sub=None, **kw):
        super().__init__()
        self.sub = sub
    def forward(self, x):
        return self.sub(x)

class Additive(nn.Module):
    def __init__(self, *args, post=None):
        super().__init__()
        self.sub = nn.ModuleList(args)
        self.post = None
    def forward(self, x):
        y = self.sub[0](x)
        for f in self.sub[1:]:
            y = y + f(x)
        if self.post is not None:
            y = self.post(y)
        return y

def ResnetBottleneck(d,  b, r=3, identity=None, post=None):
    return Additive(
        identity or nn.Identity(),
        nn.Sequential(
            nn.Conv2d(d, b, 1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, r, padding=r//2),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, d, 1)
        ),
        post = post or nn.BatchNorm2d(d)
    )

def ResnetBlock(d, r=3, identity=None, post=None):
    return Additive(
        identity or nn.Identity(),
        nn.Sequential(
            nn.Conv2d(d, d, r, padding=r//2),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, d, r, padding=r//2),
            nn.BatchNorm2d(d)
        ),
        post = post or nn.BatchNorm2d(d)
    )

################################################################
# ## entire OCR models
# ###############################################################

def make_lstm_ctc(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2mp(50, 3, (2, 1)),
        *conv2mp(100, 3, (2, 1)),
        *conv2mp(150, 3, 2),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def resnet_blocks(n, d, r=3):
    return [ResnetBlock(d, r) for _ in range(n)]

def make_lstm_resnet(noutput=noutput, blocksize=5):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2mp(64, 3, (2, 1)),
        *resnet_blocks(blocksize, 64),
        *conv2mp(128, 3, (2, 1)),
        *resnet_blocks(blocksize, 128),
        *conv2mp(256, 3, 2),
        *resnet_blocks(blocksize, 256),
        *conv2d(256, 3),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model


def make_ocr_resnet(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2mp(64, 3, 2),
        *resnet_blocks(5, 64),
        *conv2mp(128, 3, (2, 1)),
        *resnet_blocks(5, 128),
        *conv2mp(192, 3, 2),
        *resnet_blocks(5, 192),
        *conv2mp(256, 3, (2, 1)),
        *resnet_blocks(5, 256),
        *conv2d(512, 3),
        *project_and_conv1d(512, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_normalized(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1),
                     sizes=[None, 1, 80, None]),
        *conv2mp(50, 3, (2, 1)),
        *conv2mp(100, 3, (2, 1)),
        *conv2mp(150, 3, 2),
        layers.Reshape(0, [1, 2], 3),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", "BLD"))
    flex.shape_inference(model, (1, 1, 80, 200))
    return model

def make_conv_only(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2mp(100, 3, 2, repeat=2),
        *conv2mp(200, 3, 2, repeat=2),
        *conv2mp(300, 3, 2, repeat=2),
        *conv2d(400, 3, repeat=2),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_lstm2_ctc_words(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2mp(100, 3, 2, repeat=2),
        *conv2mp(200, 3, 2, repeat=2),
        *conv2mp(300, 3, 2, repeat=2),
        *conv2d(400, 3, repeat=2),
        flex.Lstm2(400),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_lstm_transpose(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2x(50, 3),
        *conv2x(100, 3),
        *conv2x(150, 3),
        *conv2x(200, 3),
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        flex.ConvTranspose1d(800, 1, stride=2), # <-- undo too tight spacing
        #flex.BatchNorm1d(), nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", "BLD")
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_keep(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        KeepSize(
            mode="nearest",
            dims=[3],
            sub=nn.Sequential(
                *conv2x(50, 3),
                *conv2x(100, 3),
                *conv2x(150, 3),
                layers.Fun("lambda x: x.sum(2)") # BDHW -> BDW
            )
        ),
        flex.Conv1d(500, 5, padding=2),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(200, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", "BLD")
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_unet(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *conv2d(64, 3, repeat=3),
        UnetLayer(64, sub=[
            *conv2d(128, 3, repeat=3),
            UnetLayer(128, sub=[
                *conv2d(256, 3, repeat=3),
                UnetLayer(256, sub=[
                    *conv2d(512, 3, repeat=3),
                ])
            ])
        ]),
        *conv2d(128, 3, repeat=2),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 256))
    return model

def unet_conv(d=64):
    return [
        *conv2d(d, 3, repeat=3),
        UnetLayer(2*d, sub=[
            *conv2d(2*d, 3, repeat=3),
            UnetLayer(3*d, sub=[
                *conv2d(3*d, 3, repeat=3),
                UnetLayer(4*d, sub=[
                    *conv2d(5*d, 3, repeat=3),
                ])
            ])
        ])
    ]


