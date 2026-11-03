"""MERT-v1-95M (m-a-p/MERT-v1-95M, cc-by-nc-4.0) layer-wise time-mean features for audio files.
Usage: python -I extract_mert.py <glob_pattern> <out_npz> <gpu> [max_seconds=30]
Audio -> mono -> 24 kHz; hidden states of all 13 layers averaged over time -> (N, 13, 768). Files processed in sorted order."""
import sys, glob, numpy as np, torch, soundfile as sf, librosa
from transformers import AutoModel, Wav2Vec2FeatureExtractor
pat, out, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]); maxs = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
dev = f"cuda:{gpu}"
proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
m = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(dev).eval()
files = sorted(glob.glob(pat)); F = []
for f in files:
    x, sr = sf.read(f, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != 24000: x = librosa.resample(x, orig_sr=sr, target_sr=24000)
    x = x[: int(maxs * 24000)]
    inp = proc(x, sampling_rate=24000, return_tensors="pt").to(dev)
    with torch.no_grad():
        h = m(**inp, output_hidden_states=True).hidden_states
    F.append(torch.stack(h).squeeze(1).mean(1).cpu().numpy())
np.savez(out, F=np.stack(F), files=np.array([f.split("/")[-1] for f in files]))
print("saved", out, np.stack(F).shape)
