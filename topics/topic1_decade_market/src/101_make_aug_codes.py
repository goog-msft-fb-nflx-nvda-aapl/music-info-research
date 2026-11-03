"""N-1: bandwidth-randomized training targets. For every dataset-A TRAIN clip, the Demucs accompaniment stem (24 kHz mono) is degraded in two random ways (seeded) and encoded with the EnCodec 32 kHz tokenizer of MusicGen: low-pass (Butterworth, order 6, cutoff uniform 3-11 kHz) with probability .7, additive white hiss at -50 to -35 dB re RMS with probability .5, so that bandwidth and noise floor no longer correlate with the decade label of the training target. Saves lora/codes_A_train_aug.npy (2, N, 4, T) int16 (same clip order and crop length T as lora/codes_A_train.npy).
usage: python 101_make_aug_codes.py   (GPU)"""
import os, numpy as np, torch, torchaudio.functional as AF
from scipy.signal import butter, sosfilt
from transformers import MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest, LABELS
rows = [r for r in manifest("A") if r["split"] == "train" and r["label"] in LABELS["A"]]; rng = np.random.default_rng(0)
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody", torch_dtype=torch.bfloat16).to("cuda"); T0 = np.load(os.path.join(ROOT, "lora", "codes_A_train.npy")).shape[2]; out = np.zeros((2, len(rows), 4, T0), np.int16)
for i, r in enumerate(rows):
    acc = np.load(os.path.join(ROOT, "stems", "A", r["sample_id"] + ".npz"))["accomp"].astype(np.float32)
    for v in range(2):
        y = acc.copy()
        if rng.random() < 0.7: y = sosfilt(butter(6, rng.uniform(3000, 11000), btype="low", fs=SR, output="sos"), y).astype(np.float32)
        if rng.random() < 0.5: y = y + rng.standard_normal(len(y)).astype(np.float32) * np.sqrt((y ** 2).mean()) * 10 ** (rng.uniform(-50, -35) / 20)
        x = AF.resample(torch.from_numpy(y), SR, 32000)[None, None].to("cuda").bfloat16()
        with torch.no_grad(): c = model.audio_encoder.encode(x).audio_codes[0, 0].cpu().numpy().astype(np.int16)
        out[v, i] = c[:, :T0]
    if i % 200 == 0: print(i, len(rows), flush=True)
np.save(os.path.join(ROOT, "lora", "codes_A_train_aug.npy"), out); print("saved", out.shape)
