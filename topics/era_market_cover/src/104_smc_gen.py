"""N-5b: probe-guided sequential Monte Carlo (SMC) era control, inference-time (no training). Same system as 99b_era_axis_gen.py (ctrlc era-axis adapter, three-branch guidance gm/ge, z of the decade prototype), but each (song, decade, seed) is generated with N particles that are reweighted and resampled every SEG decoded frames (50 frames = 1 s of audio): the partially decoded accompaniment (EnCodec decode of the frames available for all codebooks, 32 kHz -> 24 kHz, -23 LUFS) is scored with the nuisance-controlled era probe (MERT codec probe after LEACE erasure of style and bandwidth statistics, as 94_nuisance_steering.py 'both'); potential phi = log P(target decade | partial audio); incremental weight exp(BETA (phi_j - phi_{j-1})); multinomial resampling when ESS < N/2 (KV cache rows and token histories are permuted together). One particle is returned per (song, decade, seed). Melody conditioning is unchanged (chroma of the original mix in every particle).
usage: python 104_smc_gen.py --name ax_ctrlc --z A_eraz_mert_leacec0.0_s0 --gm 3 --ge 10 --N 8 --seg 100 --beta 5 --outdir gen_smc_ctrlc_N8 --seeds 0 1 2 --eras 1960s 1970s 1980s 1990s 2000s 2010s [--limit n] -> <outdir>/<song>_<era>_s<seed>_{acc,remix}.wav + smc_log.jsonl"""
import argparse, json, os, shutil, glob, numpy as np, torch, soundfile as sf, torchaudio.functional as AF, pyloudnorm as pyln
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, LogitsProcessor, LogitsProcessorList
from common import ROOT, SR, manifest, LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--z", required=True); ap.add_argument("--gm", type=float, default=3.0); ap.add_argument("--ge", type=float, default=10.0); ap.add_argument("--N", type=int, default=8); ap.add_argument("--seg", type=int, default=100); ap.add_argument("--first", type=int, default=0); ap.add_argument("--beta", type=float, default=5.0)
ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--eras", nargs="+", default=["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--max_new", type=int, default=1500); a = ap.parse_args()
DEC = LABELS["A"]; rows = manifest("A"); G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rowd = {r["sample_id"]: r for r in rows}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(G, "picked.txt"), os.path.join(out, "picked.txt"))
# ---- nuisance-controlled era scorer (fitted on dataset-A TRAIN only; MERT erased 'both') ----
import sys; from sklearn.decomposition import PCA; from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline; from concept_erasure import LeaceEraser
y = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rows]); tr = np.where(np.array([r["split"] for r in rows]) == "train")[0]
pj = json.load(open(f"{ROOT}/results/probe_codec_mert_A.json")); LAY = pj["conds"]["acccodec_norm"]["layer"]; E = np.load(f"{ROOT}/emb/A_acccodec_norm.npy").astype(np.float32)[:, LAY]
tags = np.load(f"{ROOT}/emb/A_dgtags.npy").astype(np.float64); st = np.nan_to_num(np.load(f"{ROOT}/emb/A_stats_acc.npy").astype(np.float64)); pca = PCA(10, random_state=0).fit(tags[tr]); Zs = pca.transform(tags); Zb = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-9)
Zboth = np.concatenate([Zs / Zs[tr].std(0), Zb], 1); er = LeaceEraser.fit(torch.tensor(E[tr], dtype=torch.float64), torch.tensor(Zboth[tr], dtype=torch.float64))
fe = lambda X: er(torch.tensor(X, dtype=torch.float64)).numpy().astype(np.float32); clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(fe(E[tr]), y[tr])
from mert_util import load_mert
mproc, mert = load_mert(); meter = pyln.Meter(SR)
def norm(w):
    if np.abs(w).max() < 1e-4: return w.astype(np.float32)
    l = meter.integrated_loudness(w.astype(np.float64)); return w.astype(np.float32) if not np.isfinite(l) else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
def era_probs(wavs24):   # list of 24 kHz mono arrays -> (n, 6) probabilities of the erased probe
    X = []
    for w in wavs24:
        inp = mproc(norm(w), sampling_rate=mproc.sampling_rate, return_tensors="pt").to("cuda")
        with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
        m = o.feature_attention_mask[..., None].float(); h = o.hidden_states[LAY]; X.append(((h * m).sum(1) / m.sum(1)).squeeze(0).cpu().numpy())
    X = np.array(X); ok = np.isfinite(X).all(1); P = np.full((len(X), 6), 1.0 / 6); P[ok] = clf.predict_proba(fe(X[ok])); return P
# ---- generator (as 99b) ----
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval()
from peft import PeftModel
model = PeftModel.from_pretrained(model, os.path.join(ROOT, "lora", a.name)).merge_and_unload().eval()
pf = os.path.join(ROOT, "lora", a.name, "prefix.txt"); P_ = int(open(pf).read()) if os.path.exists(pf) else 0; H = model.config.decoder.hidden_size; Dz = torch.load(os.path.join(ROOT, "lora", a.name, "A.pt"))["weight"].shape[1]
Amap = torch.nn.Linear(Dz, H * max(P_, 1), bias=False).cuda(); Amap.load_state_dict(torch.load(os.path.join(ROOT, "lora", a.name, "A.pt"))); Amap.eval()
Zall = np.load(os.path.join(ROOT, "emb", a.z + ".npy")).astype(np.float32); zst = np.load(os.path.join(ROOT, "lora", a.name + "_zstats.npz")); Zn = (Zall - zst["mean"]) / zst["std"]
PROTO = {d: Zn[[i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == d]].mean(0) for d in DEC}; PROTO["neutral"] = np.zeros(Zall.shape[1], np.float32)
CUR = {"z": None}
def film(mod, inp, out):
    if CUR["z"] is None: return out
    assert P_ == 0; return out + Amap(CUR["z"].float())[:, None, :].to(out.dtype)
model.enc_to_dec_proj.register_forward_hook(film)
NEUTRAL = "a song recording"; N = a.N; K = 4; STATE = {}; orig = MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation
def patched(self, inputs_tensor, model_kwargs, model_input_name, generation_config):
    nb = STATE.get("neutral"); gs = generation_config.guidance_scale
    if nb is None or gs is None or gs <= 1: return orig(self, inputs_tensor, model_kwargs, model_input_name, generation_config)
    kw_n = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in model_kwargs.items()}; kw_n["attention_mask"] = nb["attention_mask"]
    CUR["z"] = STATE["ze"]; out_e = orig(self, inputs_tensor, model_kwargs, model_input_name, generation_config); CUR["z"] = STATE["z0"]; out_n = orig(self, nb["input_ids"], kw_n, model_input_name, generation_config); CUR["z"] = None
    he = out_e["encoder_hidden_states"]; hn = out_n["encoder_hidden_states"]; assert he.shape[0] == 4 * N, he.shape
    out_e["encoder_hidden_states"] = torch.cat([he[0:N], hn[0:N], he[2 * N:4 * N]], dim=0); return out_e          # rows [era x N, neutral x N, null x 2N]
MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation = patched
class SMC(LogitsProcessor):
    def __init__(self): self.logw = torch.zeros(N, device="cuda"); self.prev_phi = None; self.nstage = 0; self.log = []; self.cache = None; self.dbg = True
    def reorder(self, ids, perm):
        idperm = torch.cat([torch.arange(N * K, device=ids.device).view(N, K)[perm].reshape(-1) + g * N * K for g in range(2)])   # token histories: (row, codebook) folded, groups [era, neutral]
        cperm = torch.cat([perm + g * N for g in range(4)]); ids.copy_(ids[idperm]); sc = self.cache.self_attention_cache   # cache rows: groups [era, neutral, null, null], no codebook fold
        for l in sc.layers: l.keys = l.keys[cperm]; l.values = l.values[cperm]
    def __call__(self, input_ids, scores):
        assert scores.shape[0] == 4 * N * K, scores.shape
        cols = input_ids.shape[1]; frames = cols - K
        if frames > 0 and frames >= a.first and frames % a.seg == 0 and frames <= a.max_new - K - a.seg:
            idsE = input_ids[:N * K].view(N, K, cols); codes = torch.stack([idsE[:, k, 1 + k:1 + k + frames] for k in range(K)], 1)   # (N, K, frames): token of codebook k, frame f sits in column f + 1 + k
            if self.dbg: print("cache shape", self.cache.self_attention_cache.layers[0].keys.shape, "ids", tuple(input_ids.shape), "codes", tuple(codes.shape), flush=True); self.dbg = False
            W = []
            for p in range(N):
                with torch.no_grad(): w = model.audio_encoder.decode(codes[p][None, None], [None]).audio_values[0, 0].float().cpu()
                W.append(AF.resample(w, 32000, SR).numpy())
            phi = torch.from_numpy(np.log(era_probs(W)[:, STATE["t"]] + 1e-6)).float().cuda()
            self.logw += a.beta * (phi - (self.prev_phi if self.prev_phi is not None else phi.mean())); self.prev_phi = phi
            w = torch.softmax(self.logw, 0); ess = float(1.0 / (w ** 2).sum()); self.log.append({"frames": frames, "ess": ess, "phi_mean": float(phi.mean()), "phi_max": float(phi.max())})
            if ess < N / 2:
                perm = torch.multinomial(w, N, replacement=True); self.reorder(input_ids, perm); self.prev_phi = phi[perm]; self.logw = torch.zeros(N, device="cuda"); self.log[-1]["resampled"] = True
        lp = torch.log_softmax(scores.float(), dim=-1).view(4 * N, K, -1); Le, Ln, L0 = lp[:N], lp[N:2 * N], lp[2 * N:3 * N]
        C = L0 + a.gm * (Ln - L0) + a.ge * (Le - Ln)
        if a.top_k > 0: v = torch.topk(C, a.top_k, dim=-1).values[..., -1:]; C = C.masked_fill(C < v, -1e9)
        tok = torch.multinomial(torch.softmax(C, dim=-1).view(N * K, -1), 1).view(N, K, 1); oh = torch.full_like(Le, -1e9).scatter(2, tok, 0.0)
        return torch.cat([oh, oh, oh, oh], 0).reshape(4 * N * K, -1).to(scores.dtype)
n = 0; logf = open(os.path.join(out, "smc_log.jsonl"), "a")
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
for s in ids:
    mix = sf.read(rowd[s]["path"], dtype="float32")[0]; zz = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = zz["vocals"].astype(np.float32), zz["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era in a.eras:
        STATE["t"] = DEC.index(era)
        for seed in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            tok = proc(text=[NEUTRAL], padding=True, return_tensors="pt"); feat = proc(audio=m32, sampling_rate=32000, text=[NEUTRAL], padding=True, return_tensors="pt")["input_features"]
            STATE["ze"] = torch.from_numpy(PROTO[era])[None].repeat(2 * N, 1).cuda(); STATE["z0"] = torch.zeros_like(STATE["ze"])
            STATE["neutral"] = {"input_ids": tok["input_ids"].repeat(2 * N, 1).to("cuda"), "attention_mask": tok["attention_mask"].repeat(2 * N, 1).to("cuda")}
            inp = {"input_ids": tok["input_ids"].repeat(2 * N, 1).to("cuda"), "attention_mask": tok["attention_mask"].repeat(2 * N, 1).to("cuda"), "input_features": feat.repeat(2 * N, 1, 1).to("cuda").half()}
            proc_smc = SMC(); proc_smc.cache = EncoderDecoderCache(DynamicCache(), DynamicCache()); torch.manual_seed(seed)
            with torch.no_grad(): o = model.generate(**inp, do_sample=True, top_k=a.top_k, guidance_scale=3.0, logits_processor=LogitsProcessorList([proc_smc]), max_new_tokens=a.max_new, past_key_values=proc_smc.cache)
            g = AF.resample(o[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; logf.write(json.dumps({"clip": s, "era": era, "seed": seed, "N": N, "stages": proc_smc.log}) + "\n"); logf.flush(); print(s, era, seed, n, flush=True)
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
