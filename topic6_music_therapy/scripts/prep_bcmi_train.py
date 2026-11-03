"""BCMI-MIdAS training sessions (OpenNeuro ds002724) -> windowed EEG with reported valence/arousal; same recording format as ds002722
but music is generated in real time (no clip identity). Usage: python -I prep_bcmi_train.py <ds002724_dir> <out_npz> [win_s=4] [fs=128]
Trial = [onset, onset+20 s] of each event with trial_type 1-9 (targeted affect, 3x3 grid); targets = trial mean of reported VA1/VA2 (samples > 0).
Output arrays include session (ses-N) and run, for within-subject leave-session-out evaluation."""
import sys, os, csv, glob, numpy as np, mne
from multiprocessing import Pool
mne.set_log_level("ERROR")
root, out = sys.argv[1], sys.argv[2]; win_s = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0; fs = int(sys.argv[4]) if len(sys.argv) > 4 else 128
TRIAL_S = 20.0; CODES = ["LVLA", "NVLA", "HVLA", "LVNA", "NVNA", "HVNA", "LVHA", "NVHA", "HVHA"]

def proc(f):
    sub = "sub-" + f.split("/sub-")[1].split("/")[0]; ses = f.split("/ses-")[1].split("/")[0]; run = f.split("task-run")[1].split("_")[0]
    raw = mne.io.read_raw_edf(f, preload=True); names = [c for c in raw.ch_names if c not in ("GSR", "ECG", "VA1", "VA2", "VAtarg")]
    lab = raw.copy().pick(["VA1", "VA2"]).get_data() * 1e6; sf0 = raw.info["sfreq"]; raw.pick(names); raw.filter(1.0, 45.0); raw.resample(fs)
    x = raw.get_data().astype(np.float32) * 1e6
    ev = [e for e in csv.DictReader(open(f.replace("_eeg.edf", "_events.tsv"), encoding="utf-8-sig"), delimiter="\t") if int(e["trial_type"]) <= 9]
    Xw, mw = [], []
    for ti, e in enumerate(ev):
        o = float(e["onset"]); a, b = int(o * sf0), int((o + TRIAL_S) * sf0); v1, v2 = lab[0, a:b], lab[1, a:b]
        if (v1 > 0).sum() < 100 or (v2 > 0).sum() < 100: continue
        xt = x[:, int(o * fs):int((o + TRIAL_S) * fs)]
        if xt.shape[1] < int(TRIAL_S * fs): continue
        xt = (xt - xt.mean(1, keepdims=True)) / (xt.std(1, keepdims=True) + 1e-6); n = int(win_s * fs); tv, ta = float(v1[v1 > 0].mean()), float(v2[v2 > 0].mean())
        for k in range(int(TRIAL_S // win_s)):
            Xw.append(xt[:, k * n:(k + 1) * n]); mw.append((sub, ses, f"{ses}_{run}_{ti}", k, tv, ta, CODES[int(e["trial_type"]) - 1]))
    return names, Xw, mw

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "eeg", "*_eeg.edf")))
    with Pool(32) as pool: res = [r for r in pool.map(proc, files, chunksize=1) if r is not None]
    chn = res[0][0]; assert all(r[0] == chn for r in res)
    X = np.stack([x for r in res for x in r[1]]); m = list(zip(*[t for r in res for t in r[2]]))
    np.savez_compressed(out, X=X, sub=np.array(m[0]), ses=np.array(m[1]), trial=np.array(m[2]), win=np.array(m[3]), valence=np.array(m[4], np.float32),
                        arousal=np.array(m[5], np.float32), cat=np.array(m[6]), chn=np.array(chn), fs=fs, win_s=win_s)
    print("saved", out, X.shape, "recordings", len(res), "trials", len(set(zip(m[0], m[2]))), "subjects", len(set(m[0])))
