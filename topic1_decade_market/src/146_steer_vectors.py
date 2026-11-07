"""N-8 training-free era steering of frozen MusicGen-Melody (difference-in-means activation steering, as the steering-vector papers of SURVEY5): for each decade d and each decoder layer l, v[d, l] = mean over TRAIN clips of decade d and positions of (hidden state under the decade prompt - hidden state under the neutral prompt), teacher-forced on the clip's own accompaniment codes with the clip's chroma (same inputs as 98_era_axis_lora.py, no adapter). Output steer/<name>.npy with shape (6, layers, hidden).
usage (from ~/era_market, one GPU): python src/146_steer_vectors.py --name dm64 --n 64"""
import argparse, os, sys, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, "src")
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest, LABELS
import prompts_era
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="dm64"); ap.add_argument("--n", type=int, default=64); ap.add_argument("--crop", type=int, default=500); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
cfg = model.config.decoder; K = cfg.num_codebooks; PAD = cfg.pad_token_id; BOS = cfg.bos_token_id
rows = [r for r in manifest("A") if r["split"] == "train" and r["label"] in LABELS["A"]]; codes = np.load(os.path.join(ROOT, "lora", "codes_A_train.npy")).astype(np.int64)
layers = model.decoder.model.decoder.layers; ACC = {"sum": None, "n": 0}
def hook_for(l):
    def h(mod, inp, out):
        x = (out[0] if isinstance(out, tuple) else out).float().mean((0, 1)); ACC["cur"][l] = x
    return h
for l, ly in enumerate(layers): ly.register_forward_hook(hook_for(l))
V = np.zeros((6, len(layers), cfg.hidden_size), np.float32)
for k, d in enumerate(LABELS["A"]):
    idx = [i for i, r in enumerate(rows) if r["label"] == d]; idx = list(rng.choice(idx, min(a.n, len(idx)), replace=False)); diffs = []
    for i in idx:
        r = rows[i]; s = int(rng.integers(0, codes.shape[2] - a.crop)); tgt = torch.from_numpy(codes[i:i + 1, :, s:s + a.crop]).cuda()
        m = AF.resample(torch.from_numpy(sf.read(r["path"], dtype="float32")[0]), SR, 32000).numpy(); m = m[int(s / 50 * 32000):int((s + a.crop) / 50 * 32000)]
        B, _, T = tgt.shape; dd = torch.full((B, K, T + K - 1), PAD, device="cuda", dtype=torch.long)
        for kk in range(K): dd[:, kk, kk:kk + T] = tgt[:, kk]
        dec_in = torch.cat([torch.full((B, K, 1), BOS, device="cuda", dtype=torch.long), dd[:, :, :-1]], 2).reshape(B * K, -1); H = {}
        for tag, text in (("era", prompts_era.ERAS[d]), ("neu", prompts_era.ERAS["neutral"])):
            inp = proc(text=[text], audio=[m], sampling_rate=32000, padding=True, return_tensors="pt").to("cuda"); inp = {kk: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for kk, v in inp.items()}
            ACC["cur"] = {}
            with torch.no_grad(): model(**inp, decoder_input_ids=dec_in)
            H[tag] = torch.stack([ACC["cur"][l] for l in range(len(layers))]).cpu().numpy()
        diffs.append(H["era"] - H["neu"])
    V[k] = np.mean(diffs, 0); print(d, len(idx), "mean |v| per layer", np.linalg.norm(V[k], axis=-1).round(2)[[0, 6, 12, 18, 23]], flush=True)
os.makedirs(os.path.join(ROOT, "steer"), exist_ok=True); np.save(os.path.join(ROOT, "steer", a.name + ".npy"), V); print("saved", V.shape)
