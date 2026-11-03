"""Frozen pretrained EEG foundation-model features for MET windows.
Usage: python -I extract_encoder.py <windows200.npz (uv100)> <out_npz> <cbramod|labram> <gpu>
Input windows: (N, C, 800) at 200 Hz in microvolt/100 -> (N, C, 4, 200) patches (1 s each).
cbramod: encoder output (N, C, 4, 200) -> per-channel mean over patches (N, C*200), plus global mean/std.
labram : patch tokens (N, C*4, 200) -> per-channel mean over patches (N, C*200) and global mean (200), cls token (200).
Channel names are mapped to the LaBraM 10-20 list (upper case); unmapped channels are dropped for LaBraM only."""
import sys, os, numpy as np, torch
src = os.path.expanduser("~/music_therapy/src")
path, out, which, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
dev = f"cuda:{gpu}"
z = np.load(path)
keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]
names = [str(z["chn"][i]).upper() for i in keep]
X = np.ascontiguousarray(z["X"][:, keep])
N, C, T = X.shape
assert T == 800 and int(z["fs"]) == 200
feats = {}
if which == "cbramod":
    sys.path.insert(0, os.path.join(src, "CBraMod"))
    from models.cbramod import CBraMod
    m = CBraMod().to(dev)
    m.load_state_dict(torch.load(os.path.join(src, "CBraMod/pretrained_weights/pretrained_weights.pth"), map_location=dev))
    m.proj_out = torch.nn.Identity(); m.eval()
    ch, g = [], []
    with torch.no_grad():
        for i in range(0, N, 64):
            x = torch.from_numpy(X[i:i + 64]).to(dev).reshape(-1, C, 4, 200)
            o = m(x)                      # (B, C, 4, 200)
            ch.append(o.mean(2).flatten(1).cpu().numpy())
            g.append(torch.cat([o.mean((1, 2)), o.std((1, 2))], 1).cpu().numpy())
    feats["ch_mean"] = np.concatenate(ch); feats["glob"] = np.concatenate(g)
else:
    sys.path.insert(0, os.path.join(src, "LaBraM"))
    import types
    for mod in ('pyhealth', 'pyhealth.metrics'):   # LaBraM utils imports pyhealth only for training metrics
        sys.modules[mod] = types.ModuleType(mod)
    sys.modules['pyhealth.metrics'].binary_metrics_fn = sys.modules['pyhealth.metrics'].multiclass_metrics_fn = None
    import utils as lutils
    from modeling_finetune import labram_base_patch200_200
    std = [c.upper() for c in lutils.standard_1020]
    ok = [i for i, n in enumerate(names) if n in std]
    print("labram mapped channels", len(ok), "of", C, "unmapped:", [names[i] for i in range(C) if i not in ok], flush=True)
    chans = [0] + [std.index(names[i]) + 1 for i in ok]
    m = labram_base_patch200_200(pretrained=False, num_classes=0, use_mean_pooling=False, init_values=0.1,
                                 qkv_bias=False, drop_path_rate=0.0)
    ck = torch.load(os.path.join(src, "LaBraM/checkpoints/labram-base.pth"), map_location="cpu", weights_only=False)  # official LaBraM checkpoint
    sd = ck.get("model", ck.get("module", ck))
    sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")} or sd
    print(m.load_state_dict(sd, strict=False), flush=True)
    m = m.to(dev).eval()
    ch, gm, cl = [], [], []
    with torch.no_grad():
        for i in range(0, N, 64):
            x = torch.from_numpy(X[i:i + 64][:, ok]).to(dev).reshape(-1, len(ok), 4, 200)
            tok = m.forward_features(x, input_chans=chans, return_all_tokens=True)   # (B, 1+C*4, 200)
            cl.append(tok[:, 0].cpu().numpy())
            p = tok[:, 1:].reshape(x.shape[0], len(ok), 4, 200)
            ch.append(p.mean(2).flatten(1).cpu().numpy()); gm.append(p.mean((1, 2)).cpu().numpy())
    feats["ch_mean"] = np.concatenate(ch); feats["glob"] = np.concatenate(gm); feats["cls"] = np.concatenate(cl)
np.savez_compressed(out, sub=z["sub"], song=z["song"], valence=z["valence"], arousal=z["arousal"], **feats)
print("saved", out, {k: v.shape for k, v in feats.items()})
