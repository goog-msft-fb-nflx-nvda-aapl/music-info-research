"""MET (OpenNeuro ds008701) -> windowed EEG arrays.
Usage: python -I prep_met.py <ds008701_dir> <out_npz> [win_s=4] [fs=128]
Per recording: read .set (66 EEG + EOG, 1 kHz), drop EOG, band-pass 1-45 Hz, resample to fs,
keep the conservative analysis window 6-124 s of the 130 s recording (README), drop windows that
overlap segments annotated 'bad' in events.tsv, cut non-overlapping windows of win_s seconds,
per-recording per-channel z-score (statistics from that recording only; no label information).
Output arrays: X (n_win, n_ch, win_s*fs) float32, sub, song, win (index), valence, arousal, like, fam,
cat (song category), plus channel names.
"""
import sys, os, csv, glob, numpy as np, mne
mne.set_log_level("ERROR")
root, out = sys.argv[1], sys.argv[2]
win_s = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
fs = int(sys.argv[4]) if len(sys.argv) > 4 else 128
T0, T1 = 6.0, 124.0
ratings = {}
for r in csv.DictReader(open(os.path.join(root, "ratings.tsv"), encoding="utf-8-sig"), delimiter="\t"):
    ratings[(r["participant_id"], r["song_id"])] = r
def proc(f):
    sub = "sub-" + f.split("/sub-")[1].split("/")[0]
    song = "t" + f.split("ses-t")[1].split("/")[0]
    r = ratings.get((sub, song))
    if r is None or r["Remark"].strip():
        return None
    try:
        raw = mne.io.read_raw_eeglab(f, preload=True)
    except Exception as e:
        print("SKIP unreadable", f, repr(e)[:80], flush=True)
        return None
    raw.pick(picks="eeg")
    ev = os.path.join(os.path.dirname(f), os.path.basename(f).replace("_eeg.set", "_events.tsv"))
    bad = []
    if os.path.exists(ev):
        for e in csv.DictReader(open(ev, encoding="utf-8-sig"), delimiter="\t"):
            if e["trial_type"] == "bad":
                bad.append((float(e["onset"]), float(e["onset"]) + float(e["duration"])))
    raw.filter(1.0, 45.0, verbose=False)
    raw.resample(fs)
    x = raw.get_data().astype(np.float32) * 1e6
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    n = int(win_s * fs)
    Xw, mw = [], []
    k = 0
    t = T0
    while t + win_s <= T1 + 1e-9:
        if not any(b0 < t + win_s and b1 > t for b0, b1 in bad):
            i0 = int(round(t * fs))
            Xw.append(x[:, i0:i0 + n])
            mw.append((sub, song, k, float(r["Valence"]), float(r["Arousal"]),
                       float(r["Likeability"]), float(r["Familiarity"]), r["Song Catergory"]))
        k += 1
        t += win_s
    return raw.ch_names, Xw, mw


if __name__ == "__main__":
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "eeg", "*_eeg.set")))
    with Pool(32) as pool:
        res = [r for r in pool.map(proc, files, chunksize=1) if r is not None]
    chn = res[0][0]
    assert all(r[0] == chn for r in res), "channel order differs between recordings"
    Xs = [x for r in res for x in r[1]]
    meta = [m for r in res for m in r[2]]
    X = np.stack(Xs)
    m = list(zip(*meta))
    np.savez_compressed(out, X=X, sub=np.array(m[0]), song=np.array(m[1]), win=np.array(m[2]),
                        valence=np.array(m[3], np.float32), arousal=np.array(m[4], np.float32),
                        like=np.array(m[5], np.float32), fam=np.array(m[6], np.float32),
                        cat=np.array(m[7]), chn=np.array(chn), fs=fs, win_s=win_s)
    print("saved", out, X.shape, "recordings used", len(res))
