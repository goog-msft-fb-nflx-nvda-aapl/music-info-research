"""Wrapper for the ACE-Step 1.5 Side-Step CLI (`acestep.training_v2.cli.train_fixed`) on the server: the installed torchcodec (built for CUDA 13) cannot be imported next to torch 2.10+cu128, so `torchaudio.load` fails; this wrapper replaces `torchaudio.load` by a soundfile-based loader (same return convention: float32 tensor (channels, samples), sample rate) and then runs the CLI module with the given arguments. usage (from ext/ACE-Step-1.5): python 136_run_trainer.py <cli arguments>"""
import os, sys, runpy, numpy as np, torch, torchaudio, soundfile as sf
sys.path.insert(0, os.getcwd())
def _load(path, *a, **k):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True); return torch.from_numpy(np.ascontiguousarray(x.T)), sr
torchaudio.load = _load
sys.argv = ["train_fixed"] + sys.argv[1:]; runpy.run_module("acestep.training_v2.cli.train_fixed", run_name="__main__")
