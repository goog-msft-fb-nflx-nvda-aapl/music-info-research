"""Builds the demo clips of the adapter system for the project page: generated accompaniment only (no vocals, no dataset audio), seed 0, seconds 9-21 of each 30 s clip, loudness-normalised to -23 LUFS, mp3 128 kbit/s, opaque hashed file names.
System: LoRA v1 + three-branch guidance (gm 3, ge 10); neutral = the same adapter's neutral prompt. usage: python 93_build_demo_clips.py <dir with <clip>_<era>_s0_acc.wav> <out dir docs/era_market/audio_lora> -> prints the JSON list for the page."""
import hashlib, json, os, subprocess, sys, numpy as np, soundfile as sf, pyloudnorm as pn
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
E = ["neutral", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
ids = sorted({f.split("_s0_acc")[0].rsplit("_", 1)[0] for f in os.listdir(src) if f.endswith("_s0_acc.wav")}); songs = []
for c in ids:
    row = {}
    for e in E:
        y, sr = sf.read(f"{src}/{c}_{e}_s0_acc.wav", dtype="float32"); y = y if y.ndim == 1 else y.mean(1); y = y[9 * sr:21 * sr]
        y = pn.normalize.loudness(y, pn.Meter(sr).integrated_loudness(y), -23.0); y = y / max(1.0, np.abs(y).max() / 0.97)
        h = hashlib.sha1(f"era-lora-demo-{c}-{e}".encode()).hexdigest()[:10]; tmp = f"{out}/{h}.wav"; sf.write(tmp, y, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-b:a", "128k", f"{out}/{h}.mp3"], check=True); os.remove(tmp); row[e] = f"audio_lora/{h}.mp3"
    songs.append(row)
json.dump(songs, open(out + "/../audio_lora_index.json", "w")); print(len(songs), "songs x", len(E), "clips")
