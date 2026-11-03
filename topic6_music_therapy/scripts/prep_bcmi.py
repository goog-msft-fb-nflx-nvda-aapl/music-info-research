"""BCMI-MIdAS calibration set (OpenNeuro ds002722) -> windowed EEG with reported valence/arousal.
Usage: python -I prep_bcmi.py <ds002722_dir> <out_npz> [win_s=4] [fs=128] [norm=z|uv100]
Each run EDF (1 kHz) holds 32 EEG channels, GSR, ECG and three label channels (microvolt-scaled in the EDF):
VA1 = reported valence, VA2 = reported arousal (continuous, 0 outside trials), VAtarg = targeted affect code.
events.tsv: per trial a target code (1-9, 3x3 grid LVLA..HVHA) and a music code 101-208 (-> stimuli wav, list in events.json).
Trial window = [onset, onset + 20 s] (events duration is 21 s). Targets per window: trial mean of VA1/VA2 over samples != 0
(trial-level, 'valence'/'arousal') and the window mean ('val_win'/'aro_win')."""
import sys, os, csv, glob, json, numpy as np, mne
from multiprocessing import Pool
mne.set_log_level("ERROR")
root, out = sys.argv[1], sys.argv[2]
win_s = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
fs = int(sys.argv[4]) if len(sys.argv) > 4 else 128
norm = sys.argv[5] if len(sys.argv) > 5 else "z"
TRIAL_S = 20.0
CODES = ["LVLA", "NVLA", "HVLA", "LVNA", "NVNA", "HVNA", "LVHA", "NVHA", "HVHA"]
stim = json.load(open(os.path.join(root, "sub-01/eeg/sub-01_task-run1_events.json"), encoding="utf-8-sig"))["101-209"]["Description"]
import re
STIM = re.findall(r"'([a-z]{4}\d+\.wav)'", stim)


def proc(f):
    sub = "sub-" + f.split("/sub-")[1].split("/")[0]
    run = f.split("task-run")[1].split("_")[0]
    raw = mne.io.read_raw_edf(f, preload=True)
    eeg_names = [c for c in raw.ch_names if c not in ("GSR", "ECG", "VA1", "VA2", "VAtarg")]
    lab = raw.copy().pick(["VA1", "VA2"]).get_data() * 1e6
    sf0 = raw.info["sfreq"]
    raw.pick(eeg_names)
    if norm == "uv100":
        raw.filter(0.5, 75.0); raw.notch_filter([50.0, 60.0]); raw.resample(fs)
        x = raw.get_data().astype(np.float32) * 1e6 / 100.0
    else:
        raw.filter(1.0, 45.0); raw.resample(fs)
        x = raw.get_data().astype(np.float32) * 1e6
    ev = list(csv.DictReader(open(f.replace("_eeg.edf", "_events.tsv"), encoding="utf-8-sig"), delimiter="\t"))
    trials = {}
    for e in ev:
        c, o = int(e["trial_type"]), float(e["onset"])
        d = trials.setdefault(round(o, 2), {})
        if c <= 9: d["target"] = c
        elif 101 <= c <= 208: d["music"] = c - 100
    Xw, mw = [], []
    for ti, (o, d) in enumerate(sorted(trials.items())):
        if "target" not in d or "music" not in d: continue
        a, b = int(o * sf0), int((o + TRIAL_S) * sf0)
        v1, v2 = lab[0, a:b], lab[1, a:b]
        m1, m2 = v1 > 0, v2 > 0
        if m1.sum() < 100 or m2.sum() < 100: continue
        tv, ta = float(v1[m1].mean()), float(v2[m2].mean())
        xt = x[:, int(o * fs):int((o + TRIAL_S) * fs)]
        xt = (xt - xt.mean(1, keepdims=True)) / (xt.std(1, keepdims=True) + 1e-6) if norm == "z" else xt
        n = int(win_s * fs)
        for k in range(int(TRIAL_S // win_s)):
            wv, wa = v1[int(k * win_s * sf0):int((k + 1) * win_s * sf0)], v2[int(k * win_s * sf0):int((k + 1) * win_s * sf0)]
            Xw.append(xt[:, k * n:(k + 1) * n])
            mw.append((sub, STIM[d["music"] - 1], f"{run}_{ti}", k, tv, ta,
                       float(wv[wv > 0].mean()) if (wv > 0).any() else tv, float(wa[wa > 0].mean()) if (wa > 0).any() else ta,
                       CODES[d["target"] - 1]))
    return eeg_names, Xw, mw


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(root, "sub-*", "eeg", "*_eeg.edf")))
    with Pool(32) as pool:
        res = [r for r in pool.map(proc, files, chunksize=1) if r is not None]
    chn = res[0][0]; assert all(r[0] == chn for r in res)
    X = np.stack([x for r in res for x in r[1]]); m = list(zip(*[t for r in res for t in r[2]]))
    np.savez_compressed(out, X=X, sub=np.array(m[0]), song=np.array(m[1]), trial=np.array(m[2]), win=np.array(m[3]),
                        valence=np.array(m[4], np.float32), arousal=np.array(m[5], np.float32),
                        val_win=np.array(m[6], np.float32), aro_win=np.array(m[7], np.float32), cat=np.array(m[8]),
                        chn=np.array(chn), fs=fs, win_s=win_s)
    print("saved", out, X.shape, "recordings", len(res), "trials", len(set(zip(m[0], m[2]))))
