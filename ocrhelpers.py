import numpy as np
from numpy import *
from scipy import ndimage as ndi
import torch
from functools import wraps
import editdistance
from torchmore import flex, layers
from torch import optim, nn
import torch.nn.functional as F
import sys, os
import time
import IPython
import logging

def asnp(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    else:
        assert isinstance(a, np.ndarray)
        return a

def method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        return func
    return decorator

def ctc_decode(probs, sigma=1.0, threshold=0.7, kind=None, full=False):
    probs = asnp(probs)
    assert (abs(probs.sum(1)-1) < 1e-4).all(), \
        "input not normalized; did you apply .softmax()?"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:,newaxis]
    labels, n = ndi.label(probs[:,0]<threshold)
    mask = tile(labels[:,newaxis], (1, probs.shape[1]))
    mask[:,0] = 0
    maxima = ndi.maximum_position(probs, mask, arange(1, amax(mask)+1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]
    
def pack_for_ctc(seqs):
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)

def collate4ocr(samples):
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension()==2 else im for im in images]
    w, h, d = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), w, h, d), dtype=torch.float)
    for i, im in enumerate(images):
        w, h, d = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        result[i, :w, :h, :d] = im
    allwidths = torch.tensor([im.shape[1] for im in images]).long()
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (result, allseqs, allwidths, alllens)

def collate4images(samples):
    images, targets = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension()==2 else im for im in images]
    w, h, d = map(max, zip(*[x.shape for x in images]))
    imresult = torch.zeros((len(images), w, h, d), dtype=torch.float)
    for i, im in enumerate(images):
        w, h, d = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        imresult[i, :w, :h, :d] = im
    w, h = map(max, zip(*[x.shape for x in targets]))
    tresult = torch.zeros((len(targets), w, h), dtype=torch.long)
    for i, t in enumerate(targets):
        w, h = t.shape
        tresult[i, :w, :h] = t
    return imresult, tresult

def collate_lines(samples):
    images, seqs = zip(*samples)
    inputs = layers.reorder(batch_images(images), "BHWD", "BDHW")
    b, d, h, w = inputs.size()
    assert b<=128 and d in [1, 3]
    inputs.order = "BDHW"
    return inputs, seqs

def shape_inference(model, shape):
    shape = tuple(shape)
    with torch.no_grad():
        model(torch.zeros(shape)).shape
    flex.flex_freeze(model)

freeze = shape_inference

def model_device(model):
    return next(model.parameters()).device

device = None

class LineTrainer(object):
    def __init__(self, model, *, log_softmax=True, lr=1e-4, every=3.0, device=device):
        self.model = model
        self.device = model_device(model) if device is None else device
        self.ctc_loss = nn.CTCLoss()
        self.every = every
        self.losses = []
        self.set_lr(lr)
        self.log_softmax = log_softmax
        self.clip_gradient = 1.0
        self.max_batch_size = 64
        self.batch_size = -1
        self.charset = None
        self.loss_scale = 1.0
        self.maxcount = int(os.environ.get("maxcount", 999999999))
        self.smoketest = int(os.environ.get("smoketest", 0))
        self.savedir = os.environ.get("savedir", "./models")
        # smoketests with short epochs/training
    def set_lr(self, lr, momentum=0.9):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
    def info(self):
        print(model, file=sys.stderr)
        print(self.shape, "->", self.output_shape, file=sys.stderr)
    def train_batch(self, inputs, targets, ilens, tlens):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model.forward(inputs.to(self.device))
        assert inputs.size(0) == outputs.size(0)
        olens = torch.full((outputs.size(0),), outputs.size(-1)).long()
        if hasattr(self, "compute_loss2"):
            loss = self.compute_loss2(inputs, outputs, targets, ilens, olens, tlens)
        else:
            loss = self.compute_loss(outputs, targets, olens, tlens)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (inputs, targets, ilens, tlens, outputs)
        return loss.detach().item()
    def compute_loss(self, outputs, targets, olens, tlens):
        b, l, d = outputs.size()
        assert outputs.size(0) == tlens.size(0)
        outputs = outputs.permute(1, 0, 2).cpu() # BLD -> LBD
        if self.log_softmax: outputs = outputs.log_softmax(2)
        lplens = torch.full((b,), l).long()
        # CTCLoss requires LBD
        loss = self.ctc_loss(outputs, targets, lplens, tlens)
        return loss
    def train(self, loader, epochs=1, start_epoch=0, total=None, cont=False, every=None):
        self.every = every or self.every
        if "force_epochs" in os.environ:
            epochs = int(os.environ.get("force_epochs"))
        logging.info(f"started training {self.model}")
        if not cont:
            self.losses = []
        self.last_display = time.time()
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            self.count = 0
            for images, targets, ilens, tlens in loader:
                if self.smoketest>0:
                    print(f"count: {self.count}", file=sys.stderr)
                else:
                    self.report()
                loss = self.train_batch(images, targets, ilens, tlens)
                self.losses.append(float(loss))
                self.count += 1
                if len(self.losses) >= self.maxcount:
                    break
                if self.smoketest>0 and self.count>self.smoketest:
                    print(f"smoketest finish", file=sys.stderr)
                    os.system("touch _smoketest")
                    time.sleep(1)
                    raise Exception("smoketest finished")
            if len(self.losses) >= self.maxcount: break
            self.save_epoch(epoch)
            logging.info(f"epoch {epoch} loss {mean(self.losses[-100:])}")
        self.report_end()
    def save_epoch(self, epoch):
        if not hasattr(self.model, "model_name"): return
        if not self.savedir or self.savedir=="": return
        base = self.model.model_name
        ierr = int(1e6*mean(self.losses[-100])*self.loss_scale)
        ierr = min(999999999, ierr)
        loss = "%09d" % ierr
        epoch = "%03d"%epoch
        fname = f"{self.savedir}/{base}-{epoch}-{loss}.pth"
        print(f"saving {fname}", file=sys.stderr)
        torch.save(self.model.state_dict(), fname)       
    def load(self, fname):
        print(f"loading {fname}", file=sys.stderr)
        self.model.load_state_dict(torch.load(fname))
    def load_best(self):
        assert hasattr(self.model, "model_name")
        pattern = f"{self.savedir}/{model.model_name}-*.pth"
        files = glob.glob(pattern)
        assert len(files)>0, f"no {pattern} found"
        def lossof(fname): 
            return fname.split(".")[-2].split("-")[-1]
        files = sort(files, key=lossof)
        fname = files[-1]
        self.load(fname)
    def report0(self):
        avgloss = mean(self.losses[-100:]) if len(self.losses)>0 else 0.0
        print(f"{self.epoch:3d} {self.count:9d} {avgloss:10.4f}", " "*10, file=sys.stderr, end="\r", flush=True)
    def report_end(self):
        if int(os.environ.get("noreport", 0)): return
        from IPython import display
        display.clear_output(wait=True)
    def report_outputs(self, ax, outputs):
        pred = outputs[0].detach().cpu().softmax(1).numpy()
        for i in range(pred.shape[1]):
            ax.plot(pred[:,i])
    def report(self):
        if int(os.environ.get("noreport", 0)): return
        if time.time()-self.last_display < self.every: return
        self.last_display = time.time()
        import matplotlib.pyplot as plt
        from IPython import display
        plt.close("all")
        fig = plt.figure(figsize=(10, 8))
        fig.clf()
        for i in range(3): fig.add_subplot(3, 1, i+1)
        ax1, ax2, ax3 = fig.get_axes()
        inputs, targets, ilens, tlens, outputs = self.last_batch
        ax1.set_title(f"{self.epoch} {self.count}")
        ax1.imshow(inputs[0].detach().cpu().numpy()[0,:,:])
        self.report_outputs(ax2, outputs)
        losses = ndi.gaussian_filter(self.losses, 10.0)
        losses = losses[::10]
        losses = ndi.gaussian_filter(losses, 10.0)
        ax3.plot(losses)
        ax3.set_ylim((0.9*amin(losses), median(losses)*3))
        display.clear_output(wait=True)
        display.display(fig)
    def probs_batch(self, inputs, ilens=None):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(2)
    def predict_batch(self, inputs, ilens=None, **kw):
        probs = self.probs_batch(inputs, ilens)
        result = [ctc_decode(p, **kw) for p in probs]
        return result
    def errors(self, loader):
        total = 0
        errors = 0
        for inputs, targets, ilens, tlens in loader:
            predictions = self.predict_batch(inputs)
            start = 0
            for p, l in zip(predictions, tlens):
                t = targets[start:start+l].tolist()
                errors += editdistance.distance(p, t)
                total += len(t)
                start += l
        return errors, total

class ImageTrainer(object):
    def __init__(self, model, *, lr=1e-4, every=3.0, device=device, loss=nn.NLLLoss()):
        self.model = model
        self.device = model_device(model) if device is None else device
        self.loss2d = loss
        self.every = every
        self.losses = []
        self.set_lr(lr)
        self.clip_gradient = 1.0
        self.max_batch_size = 64
        self.batch_size = -1
        self.charset = None
        self.loss_scale = 1.0
        self.maxcount = int(os.environ.get("maxcount", 999999999))
        self.smoketest = int(os.environ.get("smoketest", 0))
        self.savedir = os.environ.get("savedir", "./models")
        # smoketests with short epochs/training
    def set_lr(self, lr, momentum=0.9):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
    def train_batch(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model.forward(inputs)
        loss = self.compute_loss(inputs, outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (inputs, targets, outputs)
        return loss.detach().item()
    def compute_loss(self, inputs, outputs, targets):
        assert outputs.ndimension()==4 and targets.ndimension()==3, (outputs.size(), targets.size())
        assert outputs.size()[-2:] == targets.size()[-2:], (outputs.size(), targets.size())
        probs = F.log_softmax(outputs, dim=1)
        self.last_probs = F.softmax(outputs, dim=1)
        loss = self.loss2d(probs, targets.to(probs.device))
        return loss
    def train(self, loader, epochs=1, total=None, cont=False):
        logging.info(f"started training {self.model}")
        if not cont:
            self.losses = []
        self.last_display = time.time()
        for epoch in range(epochs):
            self.epoch = epoch
            self.count = 0
            for inputs, targets in loader:
                if self.smoketest>0:
                    print(f"count: {self.count}", file=sys.stderr)
                else:
                    self.report()
                loss = self.train_batch(inputs, targets)
                self.losses.append(loss)
                self.count += 1
                if len(self.losses) >= self.maxcount:
                    break
                if self.smoketest>0 and self.count>self.smoketest:
                    print(f"smoketest finish", file=sys.stderr)
                    os.system("touch _smoketest")
                    time.sleep(1)
                    raise Exception("smoketest finished")
            if len(self.losses) >= self.maxcount: break
            self.save_epoch(epoch)
            logging.info(f"epoch {epoch} loss {mean(self.losses[-100:])}")
        self.report_end()
    def save_epoch(self, epoch):
        if not hasattr(self.model, "model_name"): return
        if not self.savedir or self.savedir=="": return
        base = self.model.model_name
        ierr = int(1e6*mean(self.losses[-100])*self.loss_scale)
        ierr = min(999999999, ierr)
        loss = "%09d" % ierr
        epoch = "%03d"%epoch
        fname = f"{self.savedir}/{base}-{epoch}-{loss}.pth"
        print(f"saving {fname}", file=sys.stderr)
        torch.save(self.model.state_dict(), fname)       
    def load(self, fname):
        print(f"loading {fname}", file=sys.stderr)
        self.model.load_state_dict(torch.load(fname))
    def load_best(self):
        assert hasattr(self.model, "model_name")
        pattern = f"{self.savedir}/{model.model_name}-*.pth"
        files = glob.glob(pattern)
        assert len(files)>0, f"no {pattern} found"
        def lossof(fname): 
            return fname.split(".")[-2].split("-")[-1]
        files = sort(files, key=lossof)
        fname = files[-1]
        self.load(fname)
    def report0(self):
        avgloss = mean(self.losses[-100:]) if len(self.losses)>0 else 0.0
        print(f"{self.epoch:3d} {self.count:9d} {avgloss:10.4f}", " "*10, file=sys.stderr, end="\r", flush=True)
    def report_end(self):
        if int(os.environ.get("noreport", 0)): return
        from IPython import display
        display.clear_output(wait=True)
    def report_outputs(self, ax, outputs):
        b, d, h, w = self.last_probs.size()
        ax.imshow(asnp(self.last_probs[0,0]))
    def report(self):
        if int(os.environ.get("noreport", 0)): return
        if time.time()-self.last_display < self.every: return
        self.last_display = time.time()
        import matplotlib.pyplot as plt
        from IPython import display
        plt.close("all")
        fig = plt.figure(figsize=(10, 8))
        fig.clf()
        for i in range(3): fig.add_subplot(3, 1, i+1)
        ax1, ax2, ax3 = fig.get_axes()
        inputs, targets, outputs = self.last_batch
        ax1.set_title(f"{self.epoch} {self.count}")
        ax1.imshow(inputs[0].detach().cpu().numpy()[0,:,:])
        self.report_outputs(ax2, outputs)
        losses = ndi.gaussian_filter(self.losses, 10.0)
        losses = losses[::10]
        losses = ndi.gaussian_filter(losses, 10.0)
        ax3.plot(losses)
        ax3.set_ylim((0.9*amin(losses), median(losses)*3))
        display.clear_output(wait=True)
        display.display(fig)
    def probs_batch(self, inputs, ilens=None):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().cpu().softmax(2)
    def predict_batch(self, inputs, ilens=None, **kw):
        probs = self.probs_batch(inputs, ilens)
        result = [ctc_decode(p, **kw) for p in probs]
        return result
    def errors(self, loader):
        total = 0
        errors = 0
        for inputs, targets, ilens, tlens in loader:
            predictions = self.predict_batch(inputs)
            start = 0
            for p, l in zip(predictions, tlens):
                t = targets[start:start+l].tolist()
                errors += editdistance.distance(p, t)
                total += len(t)
                start += l
        return errors, total

import matplotlib.pyplot as plt
plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")
import scipy.ndimage as ndi

def RUN(x): print(x, ":", os.popen(x).read().strip())
    
def scale_to(a, shape):
    scales = array(a.shape, "f") / array(shape, "f")
    result = ndi.affine_transform(a, diag(scales), output_shape=shape, order=1)
    return result

def tshow(a, order, b=0, ax=None, **kw):
    ax = ax or gca()
    if set(order)==set("BHWD"):
        a = layers.reorder(a.detach().cpu(), order, "BHWD")[b].numpy()
    elif set(order)==set("HWD"):
        a = layers.reorder(a.detach().cpu(), order, "HWD").numpy()
    elif set(order)==set("HW"):
        a = layers.reorder(a.detach().cpu(), order, "HW").numpy()
    else:
        raise ValueError(f"{order}: unknown order")
    if a.shape[-1]==1: a = a[...,0]
    ax.imshow(a, **kw)
