"""Valence/arousal-conditioned LoRA fine-tuning of Stable Audio Open 1.0 (diffusers) on PMEmo chorus clips, and sampling.
Usage:
  train: python -I sao_va.py train <run_name> <gpu> <label_mode: true|shuffled|none> [steps=2000] [seed=0]
  gen  : python -I sao_va.py gen   <run_name> <gpu> [n_per_cell=6] [steps=50] [cfg=3.0]
Conditioning: encoder_hidden_states = [T5 prompt tokens ("music"), seconds_start, seconds_end] + 2 VA tokens produced by an MLP from (valence, arousal) in [0,1];
VA tokens are replaced by a learned null embedding with probability 0.15 during training (modality dropout -> classifier-free guidance on VA).
Objective: EDM/v-prediction exactly as in the diffusers CosineDPM scheduler (sigma_data=1), u ~ U(0.02,0.999), sigma = tan(u*pi/2), unweighted MSE in network-output space.
Trainable: LoRA (rank 16) on all attention projections of the DiT + VA module. Data: PMEmo chorus clips, 10 s crops, song-level 85/15 split (seed 0)."""
import sys, os, math, glob, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, soundfile as sf, librosa
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from peft import LoraConfig, inject_adapter_in_model
H = os.path.expanduser("~/music_therapy"); SR = 44100; CROP = 10 * SR; NLAT = 216
mode, run, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = f"cuda:{gpu}"
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()


class VAModule(nn.Module):
    def __init__(self, d=768, ntok=2):
        super().__init__(); self.ntok, self.d = ntok, d
        self.mlp = nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, ntok * d))
        self.null = nn.Parameter(torch.zeros(ntok, d))
    def forward(self, va, drop=None):               # va (B,2) in [0,1]; drop (B,) bool -> null tokens
        t = self.mlp((va - 0.5) * 4).view(-1, self.ntok, self.d)
        if drop is not None: t = torch.where(drop[:, None, None], self.null[None].expand_as(t), t)
        return t


def load_pipe():
    p = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32, token=tok)
    return p.to(dev)


@torch.no_grad()
def base_cond(pipe):
    pe = pipe.encode_prompt("music", dev, False)
    s, e = pipe.encode_duration(0.0, 10.0, dev, False, 1)
    return torch.cat([pe, s, e], 1), torch.cat([s, e], 2)          # (1,T+2,768), (1,1,1536)


def run_transformer(tr, x, u, enc, glob_, B):
    rot = get_1d_rotary_pos_embed(tr.config.attention_head_dim // 2, x.shape[2] + glob_.shape[1], use_real=True, repeat_interleave_real=False)
    rot = tuple(r.to(x.device) for r in rot) if isinstance(rot, tuple) else rot.to(x.device)
    return tr(x, u, encoder_hidden_states=enc, global_hidden_states=glob_.expand(B, -1, -1), rotary_embedding=rot, return_dict=False)[0]


def make_split():
    ann = pd.read_csv(f"{H}/data/pmemo/x/PMEmo2019/annotations/static_annotations.csv").set_index("musicId")
    ids = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(f"{H}/data/pmemo/x/PMEmo2019/chorus/*.mp3"))
    ids = [i for i in ids if i in ann.index]; rng = np.random.default_rng(0); perm = rng.permutation(len(ids)); nte = int(0.15 * len(ids))
    te = set(np.array(ids)[perm[:nte]].tolist())
    return ann, ids, te


def get_latents(pipe):
    path = f"{H}/data/feat/sao_lat_pmemo.pt"
    if os.path.exists(path): return torch.load(path)
    ann, ids, te = make_split(); L, V, S = [], [], []
    for i in ids:
        x, sr = sf.read(f"{H}/data/pmemo/x/PMEmo2019/chorus/{i}.mp3", dtype="float32", always_2d=True)
        if sr != SR: x = librosa.resample(x.T, orig_sr=sr, target_sr=SR).T
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        for k in range(x.shape[0] // CROP):
            seg = torch.from_numpy(x[k * CROP:(k + 1) * CROP].T).to(dev); seg = nn.functional.pad(seg, (0, NLAT * 2048 - CROP))
            with torch.no_grad(): z = pipe.vae.encode(seg[None]).latent_dist.sample()
            L.append(z[0].cpu()); V.append(ann.loc[i][["Valence(mean)", "Arousal(mean)"]].values.astype(np.float32)); S.append(i)
    d = dict(lat=torch.stack(L), va=torch.from_numpy(np.stack(V)), song=torch.tensor(S)); torch.save(d, path); return d


if mode == "train":
    label_mode = sys.argv[4]; steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2000; seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    torch.manual_seed(seed); np.random.seed(seed)
    pipe = load_pipe(); d = get_latents(pipe); _, ids, te = make_split()
    tri = torch.tensor([s.item() not in te for s in d["song"]]); lat, va, song = d["lat"][tri], d["va"][tri], d["song"][tri]
    print("train crops", len(lat), "songs", len(set(song.tolist())), flush=True)
    if label_mode == "shuffled":                               # permute labels across songs (control)
        us = sorted(set(song.tolist())); p = dict(zip(us, np.random.default_rng(seed).permutation(us))); sv = {s: va[song == s][0] for s in us}
        va = torch.stack([sv[int(p[int(s)])] for s in song])
    enc0, glob0 = base_cond(pipe); tr = pipe.transformer
    for q in list(pipe.vae.parameters()) + list(pipe.text_encoder.parameters()): q.requires_grad_(False)
    tr.requires_grad_(False); tr = inject_adapter_in_model(LoraConfig(r=16, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"]), tr); pipe.transformer = tr
    for n_, p_ in tr.named_parameters(): p_.requires_grad_("lora" in n_)
    vam = VAModule().to(dev); params = [p for p in tr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-4}, {"params": vam.parameters(), "lr": 1e-3}], weight_decay=0.01)
    print("trainable LoRA params %.2fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    lat, va = lat.to(dev), va.to(dev); B = 8; t0 = time.time(); log = []
    for step in range(steps):
        idx = torch.randint(0, len(lat), (B,), device=dev); x0 = lat[idx]; v = va[idx] if label_mode != "none" else va[idx]
        drop = torch.rand(B, device=dev) < (0.15 if label_mode != "none" else 1.0)
        u = torch.rand(B, device=dev) * 0.979 + 0.02; sig = torch.tan(u * math.pi / 2); s3 = sig[:, None, None]
        eps = torch.randn_like(x0); xt = x0 + s3 * eps; cin = 1 / torch.sqrt(s3 ** 2 + 1); cskip = 1 / (s3 ** 2 + 1); cout = -s3 / torch.sqrt(s3 ** 2 + 1)
        target = (x0 - cskip * xt) / cout
        enc = torch.cat([enc0.expand(B, -1, -1), vam(v, drop)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = run_transformer(tr, xt * cin, u, enc, glob0, B)
        loss = nn.functional.mse_loss(out.float(), target); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params + list(vam.parameters()), 1.0); opt.step(); log.append(loss.item())
        if step % 50 == 0: print(f"step {step} loss {np.mean(log[-50:]):.4f} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(f"{H}/results/gen/{run}", exist_ok=True)
    torch.save({"lora": {k: v_ for k, v_ in tr.state_dict().items() if "lora" in k}, "va": vam.state_dict(), "label_mode": label_mode, "steps": steps}, f"{H}/results/gen/{run}/ckpt.pt")
    json.dump(log, open(f"{H}/results/gen/{run}/train_loss.json", "w")); print("saved", flush=True)

else:   # gen
    n_cell = int(sys.argv[4]) if len(sys.argv) > 4 else 6; steps = int(sys.argv[5]) if len(sys.argv) > 5 else 50; cfg = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
    pipe = load_pipe(); tr = pipe.transformer; enc0, glob0 = base_cond(pipe)
    ck = torch.load(f"{H}/results/gen/{run}/ckpt.pt", map_location=dev) if run != "base" else None
    vam = VAModule().to(dev)
    if ck is not None:
        tr = inject_adapter_in_model(LoraConfig(r=16, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"]), tr); print(tr.load_state_dict(ck["lora"], strict=False).unexpected_keys[:2]); vam.load_state_dict(ck["va"])
    grid = [(v, a) for v in (0.2, 0.5, 0.8) for a in (0.2, 0.5, 0.8)]; vas = torch.tensor([g for g in grid for _ in range(n_cell)], dtype=torch.float32, device=dev); N = len(vas)
    outs = []
    for b0 in range(0, N, 18):
        va_b = vas[b0:b0 + 18]; B = len(va_b); g = torch.Generator(dev).manual_seed(1000 + b0)
        pipe.scheduler.set_timesteps(steps, device=dev); lat = torch.randn(B, 64, NLAT, device=dev, generator=g) * pipe.scheduler.init_noise_sigma
        enc_c = torch.cat([enc0.expand(B, -1, -1), vam(va_b)], 1); enc_u = torch.cat([enc0.expand(B, -1, -1), vam(va_b, torch.ones(B, dtype=torch.bool, device=dev))], 1)
        if ck is not None and ck["label_mode"] == "none": enc_c = enc_u      # baseline fine-tuned without VA: null tokens only
        for t in pipe.scheduler.timesteps:
            x_in = pipe.scheduler.scale_model_input(torch.cat([lat, lat]), t)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = run_transformer(tr, x_in, t.expand(2 * B), torch.cat([enc_c, enc_u]), glob0, 2 * B).float()
            oc, ou = o.chunk(2); lat = pipe.scheduler.step(ou + cfg * (oc - ou), t, lat).prev_sample
        with torch.no_grad(): au = pipe.vae.decode(lat).sample[:, :, :CROP].float().cpu()
        outs.append(au)
    au = torch.cat(outs).numpy(); os.makedirs(f"{H}/results/gen/{run}", exist_ok=True)
    np.savez_compressed(f"{H}/data/feat/gen_{run}{'' if cfg == 3.0 else f'_cfg{cfg:g}'}_n{n_cell}.npz" if n_cell != 6 else f"{H}/data/feat/gen_{run}{'' if cfg == 3.0 else f'_cfg{cfg:g}'}.npz", audio=au.astype(np.float16), va=vas.cpu().numpy(), sr=SR); print("generated", au.shape)
