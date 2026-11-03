"""End-to-end fine-tuning of a pretrained EEG foundation model (CBraMod or LaBraM) for valence/arousal regression with grouped cross-validation.
Usage: python -I finetune_fm.py <windows200.npz (uv100)> <out_dir> <cbramod|labram> <protos comma list e.g. loso,loso_song,strict> [ngpu=4] [epochs=6]
Windows are 4 s at 200 Hz in microvolt/100 -> (B, C, 4, 200). Head: mean over channels and patches -> LayerNorm -> Linear(200,2) on standardized targets.
Backbone lr 5e-5, head lr 1e-3, AdamW, cosine schedule, batch 128, bf16 autocast, fixed epochs (no tuning on test folds). Metrics as in met_eval.py (trial level)."""
import sys, os, json, time, types, numpy as np, torch, torch.nn as nn, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from met_eval import make_folds, trial_metrics, boot_ci
SRC = os.path.expanduser("~/music_therapy/src")
_D = {}

def load(path):
    z = np.load(path); keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]
    X = np.ascontiguousarray(z["X"][:, keep]); X = X.reshape(len(X), len(keep), 4, 200)
    return dict(X=X, sub=z["sub"], song=z["song"], Y=np.stack([z["valence"], z["arousal"]], 1).astype(np.float32), names=[str(z["chn"][i]).upper() for i in keep])

def init(path, which):
    _D.update(load(path)); _D["which"] = which

class Net(nn.Module):
    def __init__(self, which, names):
        super().__init__(); self.which = which
        if which == "cbramod":
            sys.path.insert(0, os.path.join(SRC, "CBraMod")); from models.cbramod import CBraMod
            self.enc = CBraMod(); self.enc.load_state_dict(torch.load(os.path.join(SRC, "CBraMod/pretrained_weights/pretrained_weights.pth"), map_location="cpu")); self.enc.proj_out = nn.Identity()
        else:
            sys.path.insert(0, os.path.join(SRC, "LaBraM"))
            for m in ("pyhealth", "pyhealth.metrics"): sys.modules[m] = types.ModuleType(m)
            sys.modules["pyhealth.metrics"].binary_metrics_fn = sys.modules["pyhealth.metrics"].multiclass_metrics_fn = None
            import utils as lu; from modeling_finetune import labram_base_patch200_200
            std = [c.upper() for c in lu.standard_1020]; self.chans = [0] + [std.index(n) + 1 for n in names]
            self.enc = labram_base_patch200_200(pretrained=False, num_classes=0, use_mean_pooling=False, init_values=0.1, qkv_bias=False, drop_path_rate=0.1)
            ck = torch.load(os.path.join(SRC, "LaBraM/checkpoints/labram-base.pth"), map_location="cpu", weights_only=False); sd = ck.get("model", ck)
            sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")} or sd; self.enc.load_state_dict(sd, strict=False)
        self.head = nn.Sequential(nn.LayerNorm(200), nn.Linear(200, 2))
    def forward(self, x):
        if self.which == "cbramod": h = self.enc(x).mean((1, 2))
        else: h = self.enc.forward_features(x, input_chans=self.chans, return_all_tokens=True)[:, 1:].mean(1)
        return self.head(h)

def run(job):
    gpu, proto, k, tr, te, epochs = job; dev = f"cuda:{gpu}"; torch.manual_seed(k)
    X, Y = _D["X"], _D["Y"]; m, s = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    net = Net(_D["which"], _D["names"]).to(dev)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 5e-5}, {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    idx = np.where(tr)[0]; steps = epochs * int(np.ceil(len(idx) / 128)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [5e-5, 1e-3], total_steps=steps, pct_start=0.1)
    Yt = torch.from_numpy((Y - m) / s)
    for ep in range(epochs):
        net.train(); perm = np.random.default_rng(100 * k + ep).permutation(idx)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]; xb = torch.from_numpy(X[b]).to(dev); yb = Yt[b].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): out = net(xb)
            loss = nn.functional.mse_loss(out.float(), yb); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); ti = np.where(te)[0]; P = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(ti), 256): P.append(net(torch.from_numpy(X[ti[i:i + 256]]).to(dev)).float().cpu().numpy())
    return proto, k, ti, np.concatenate(P) * s + m

if __name__ == "__main__":
    path, out_dir, which, protos = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    ngpu = int(sys.argv[5]) if len(sys.argv) > 5 else 4; epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 6; os.makedirs(out_dir, exist_ok=True)
    d = load(path); sub, song = d["sub"], d["song"]; jobs = []; n = 0
    for p in protos:
        for k, (tr, te) in enumerate(make_folds(p, sub, song)): jobs.append((n % ngpu, p, k, tr, te, epochs)); n += 1
    t0 = time.time(); ctx = mp.get_context("spawn")
    with ctx.Pool(ngpu, initializer=init, initargs=(path, which)) as pool: res = pool.map(run, jobs, chunksize=1)
    print("folds done in %.0fs" % (time.time() - t0), flush=True)
    tk = np.array([a + "|" + b for a, b in zip(sub, song)]); u, inv = np.unique(tk, return_inverse=True); ts = np.array([x.split("|")[0] for x in u])
    ty = np.stack([d["Y"][inv == i][0] for i in range(len(u))]); summ = {}
    for p in protos:
        P = np.full(d["Y"].shape, np.nan)
        for pp, k, ti, pr in res:
            if pp == p: P[ti] = pr
        g = ~np.isnan(P[:, 0]); tp = np.stack([P[inv == i].mean(0) for i in range(len(u))])
        for j, t in enumerate(("valence", "arousal")):
            m_ = trial_metrics(ty[:, j], tp[:, j], ts); m_["ci"] = boot_ci(ty[:, j], tp[:, j], ts, n=500); m_["n_trials"] = len(u); summ[f"{p}/{which}/{t}"] = m_
        np.save(os.path.join(out_dir, f"pred_{which}_{p}.npy"), tp)
    json.dump(summ, open(os.path.join(out_dir, f"summary_ft_{which}.json"), "w"), indent=1)
    for k_, v in summ.items(): print(k_, {a: round(b, 3) for a, b in v.items() if a not in ("ci", "n_trials")}, [round(x, 3) for x in v["ci"]["ccc"]])
