"""EEG -> valence/arousal on MET with grouped cross-validation.
Usage: python -I met_eval.py <windows.npz> <out_dir> <model: ridge|cnn> [gpus=4]
Protocols (all predictions are made on windows, then averaged per trial = (subject, song)):
  loso     leave-one-subject-out (20 folds); test subject unseen, songs seen
  loso_song leave-one-song-out (20 folds); test song unseen, subjects seen
  strict   5 subject groups x 4 song groups = 20 folds; test cell has unseen subjects AND unseen songs
  random   random 5-fold over windows (leaky reference, not a valid protocol)
Reference predictors (no EEG): global mean; song mean from training subjects (loso only); shuffled-label control (ridge).
Metrics per target at trial level: Pearson r, CCC, RMSE (1-9 scale), subject-level bootstrap 95% CI (1000 resamples).
"""
import sys, os, json, numpy as np, time
import torch, torch.nn as nn, torch.multiprocessing as mp

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def band_features(X, fs, dev):
    out = []
    for i in range(0, len(X), 512):
        x = torch.from_numpy(X[i:i + 512]).to(dev)
        n = x.shape[-1]
        p = torch.fft.rfft(x * torch.hann_window(n, device=dev), dim=-1).abs() ** 2
        f = torch.fft.rfftfreq(n, 1.0 / fs).to(dev)
        feats = [torch.log(p[..., (f >= lo) & (f < hi)].mean(-1) + 1e-8) for lo, hi in BANDS]
        out.append(torch.cat(feats, -1).cpu().numpy())
    return np.concatenate(out)


def ccc(a, b):
    ma, mb = a.mean(), b.mean()
    return 2 * np.mean((a - ma) * (b - mb)) / (a.var() + b.var() + (ma - mb) ** 2 + 1e-12)


def trial_metrics(y, p, sub):
    r = np.corrcoef(y, p)[0, 1] if p.std() > 0 else 0.0
    return dict(r=float(r), ccc=float(ccc(y, p)), rmse=float(np.sqrt(np.mean((y - p) ** 2))))


def boot_ci(y, p, sub, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.unique(sub)
    idx = {s: np.where(sub == s)[0] for s in subs}
    res = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs))
        ii = np.concatenate([idx[s] for s in pick])
        if p[ii].std() == 0:
            continue
        res.append(trial_metrics(y[ii], p[ii], None))
    return {k: [float(np.percentile([r[k] for r in res], 2.5)), float(np.percentile([r[k] for r in res], 97.5))]
            for k in ("r", "ccc", "rmse")}


def make_folds(proto, sub, song, seed=0):
    us, ug = np.unique(sub), np.unique(song)
    rng = np.random.default_rng(seed)
    if proto == "loso":
        return [(~(sub == s), sub == s) for s in us]
    if proto == "loso_song":
        return [(~(song == g), song == g) for g in ug]
    if proto == "strict":
        sg = np.array_split(rng.permutation(us), 5)
        gg = np.array_split(rng.permutation(ug), 4)
        return [(~(np.isin(sub, a) | np.isin(song, b)), np.isin(sub, a) & np.isin(song, b)) for a in sg for b in gg]
    if proto == "random":
        f = rng.integers(0, 5, len(sub))
        return [(f != k, f == k) for k in range(5)]
    raise ValueError(proto)


def ridge_fit_predict(Xtr, Ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return B @ W + ym


class EEGNet(nn.Module):
    def __init__(self, nch, nt, nout=2, F1=16, D=2, F2=32, drop=0.5):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (nch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
                                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n = self.b2(self.b1(torch.zeros(1, 1, nch, nt))).numel()
        self.fc = nn.Linear(n, nout)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x.unsqueeze(1))).flatten(1))


def cnn_fit_predict(Xtr, Ytr, Xte, dev, epochs=30, seed=0):
    torch.manual_seed(seed)
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    Y = torch.from_numpy(((Ytr - ym) / ys).astype(np.float32)).to(dev)
    Xt = torch.from_numpy(Xtr).to(dev)
    m = EEGNet(Xtr.shape[1], Xtr.shape[2]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.mse_loss(m(Xt[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        P = torch.cat([m(torch.from_numpy(Xte[i:i + 512]).to(dev)) for i in range(0, len(Xte), 512)]).cpu().numpy()
    return P * ys + ym


_D = None


def load_data(path, model, dev):
    z = np.load(path)
    d = {k: z[k] for k in ('sub', 'song', 'valence', 'arousal', 'cat')}
    keep = [i for i, c in enumerate(z['chn']) if str(c).upper() != 'EOG']
    X = z['X'][:, keep]
    if model == 'ridge':
        d['F'] = band_features(X, int(z['fs']), dev)
    else:
        d['X'] = np.ascontiguousarray(X)
    return d


def _init(path, model):
    global _D
    _D = load_data(path, model, 'cpu')


def run_fold(args):
    gpu, model, proto, k, tr, te = args
    d = _D
    dev = f"cuda:{gpu}"
    Y = np.stack([d["valence"], d["arousal"]], 1)
    if model == "ridge":
        F = d["F"]
        out = {f"a{a}": ridge_fit_predict(F[tr], Y[tr], F[te], a) for a in (0.1, 1.0, 10.0)}
        rng = np.random.default_rng(k)
        # shuffled-label control: permute trial labels across training trials
        trials = {}
        for i in np.where(tr)[0]:
            trials.setdefault((d["sub"][i], d["song"][i]), []).append(i)
        keys = list(trials); perm = rng.permutation(len(keys))
        Ysh = Y.copy()
        for a_, b_ in zip(keys, perm):
            for i in trials[a_]:
                Ysh[i] = Y[trials[keys[b_]][0]]
        out["shuf"] = ridge_fit_predict(F[tr], Ysh[tr], F[te], 1.0)
        return proto, k, np.where(te)[0], out
    P = cnn_fit_predict(d["X"][tr], Y[tr], d["X"][te], dev)
    return proto, k, np.where(te)[0], {"cnn": P}


def main():
    path, out_dir, model = sys.argv[1], sys.argv[2], sys.argv[3]
    ngpu = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(out_dir, exist_ok=True)
    global _D
    _D = d = load_data(path, model, "cuda:0")
    sub, song = d["sub"], d["song"]
    trial_ids = np.array([a + "|" + b for a, b in zip(sub, song)])
    tr_keys, inv = np.unique(trial_ids, return_inverse=True)
    protos = ["loso", "loso_song", "strict", "random"]
    jobs = []
    n = 0
    for proto in protos:
        for k, (tr, te) in enumerate(make_folds(proto, sub, song)):
            jobs.append((n % ngpu, model, proto, k, tr, te)); n += 1
    t0 = time.time()
    if model == "cnn":
        ctx = mp.get_context("spawn")
        with ctx.Pool(ngpu, initializer=_init, initargs=(path, model)) as pool:
            res = pool.map(run_fold, jobs, chunksize=1)
    else:
        res = [run_fold(j) for j in jobs]
    print("folds done in %.0fs" % (time.time() - t0), flush=True)
    # collect window predictions per protocol/predictor, average per trial
    summary = {}
    names = sorted({k for r in res for k in r[3]})
    for proto in protos:
        for nm in names:
            P = np.full((len(sub), 2), np.nan)
            for p_, k, idx, o in res:
                if p_ == proto and nm in o:
                    P[idx] = o[nm]
            ok = ~np.isnan(P[:, 0])
            tp = np.stack([P[ok & (inv == t)].mean(0) if (ok & (inv == t)).any() else [np.nan, np.nan] for t in range(len(tr_keys))])
            ty = np.stack([[d["valence"][inv == t][0], d["arousal"][inv == t][0]] for t in range(len(tr_keys))])
            ts = np.array([k_.split("|")[0] for k_ in tr_keys])
            g = ~np.isnan(tp[:, 0])
            for j, tgt in enumerate(("valence", "arousal")):
                m = trial_metrics(ty[g, j], tp[g, j], ts[g]); m["ci"] = boot_ci(ty[g, j], tp[g, j], ts[g])
                m["n_trials"] = int(g.sum())
                summary[f"{proto}/{nm}/{tgt}"] = m
            np.save(os.path.join(out_dir, f"pred_{model}_{proto}_{nm}.npy"), tp)
    # reference predictors without EEG
    ty = np.stack([[d["valence"][inv == t][0], d["arousal"][inv == t][0]] for t in range(len(tr_keys))])
    ts = np.array([k_.split("|")[0] for k_ in tr_keys]); tg = np.array([k_.split("|")[1] for k_ in tr_keys])
    for j, tgt in enumerate(("valence", "arousal")):
        # global mean over other subjects (loso)
        pg = np.array([ty[ts != s, j].mean() for s in ts])
        summary[f"ref/global_mean_loso/{tgt}"] = dict(trial_metrics(ty[:, j], pg, ts), n_trials=len(ts))
        # song mean from other subjects (loso): uses knowledge of the song identity, no EEG
        ps = np.array([ty[(tg == g) & (ts != s), j].mean() for s, g in zip(ts, tg)])
        m = trial_metrics(ty[:, j], ps, ts); m["ci"] = boot_ci(ty[:, j], ps, ts); m["n_trials"] = len(ts)
        summary[f"ref/song_mean_loso/{tgt}"] = m
        # subject mean from other songs (loso_song)
        pu = np.array([ty[(ts == s) & (tg != g), j].mean() for s, g in zip(ts, tg)])
        summary[f"ref/subject_mean_loso_song/{tgt}"] = dict(trial_metrics(ty[:, j], pu, ts), n_trials=len(ts))
    json.dump(summary, open(os.path.join(out_dir, f"summary_{model}.json"), "w"), indent=1)
    for k_, v in summary.items():
        print(k_, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "ci"}, v.get("ci", {}).get("ccc"))


if __name__ == "__main__":
    main()
