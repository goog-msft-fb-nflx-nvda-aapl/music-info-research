"""T-4a: timbre-only market test with FreeSVC (zero-shot multilingual singing voice conversion, github.com/freds0/free-svc, weights alefiury/free-svc; run from ~/era_market/ext/free-svc with venv_svc).
Every dataset-B validation vocal stem (102 clips, vocals from 01_separate.py) is converted to the voice of each of the six release markets: market voice = mean ECAPA2 speaker embedding of the vocal stems of K train clips of that market (first K by sample id; K=10); the source language id is the Whisper-large-v3 language group of the clip (emb/B_lid3_voc.npy), mapped to the FreeSVC language ids (unknown -> english); config `src/108_svc_config.yaml` (hf_weights/config.yaml with the speaker/content/language settings of the released checkpoint: ECAPA2 speaker encoder, SPIN content encoder, language embedding; copy to ~/era_market/svc_config.yaml; spin.ckpt and rmvpe.pt from hf_weights, rmvpe linked to models/f0_predictor/ckpt/rmvpe.pt). Melody (pitch) and content are kept by the model; only the speaker timbre changes. Output: <outdir>/<sample_id>_<market>.wav (model sampling rate).
Variant: --nearest uses, for each market, the TRAIN vocal stem of that market whose ECAPA2 embedding is nearest to the source clip's (voice-matched reference instead of the mean of K clips; write to svc_B_nn).
Control: --selfvoice converts each clip with its OWN ECAPA2 embedding (same pipeline, no timbre change; output <id>_self.wav) to separate the conversion artifacts from the timbre swap.
usage (from ~/era_market/ext/free-svc): ~/era_market/venv_svc/bin/python ~/era_market/src/108_svc_market.py --outdir ~/era_market/svc_B [--limit 2]"""
import os, sys, csv, json, argparse, numpy as np, torch, librosa
from scipy.io import wavfile
sys.path.insert(0, os.getcwd())
import utils
from models import SynthesizerTrn
from models.speaker_encoders import ECAPA2SpeakerEncoder16k
from models.f0_predictor import get_f0_predictor
ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--selfvoice", action="store_true"); ap.add_argument("--nearest", action="store_true"); ap.add_argument("--K", type=int, default=10); ap.add_argument("--hpfile", default=os.path.expanduser("~/era_market/svc_config.yaml")); ap.add_argument("--pitch-factor", type=float, default=0.9544); a = ap.parse_args()
H = os.path.expanduser("~/era_market"); MK = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
rows = list(csv.DictReader(open(f"{H}/data/ex/dataset_B/manifest.csv"))); lid = np.load(f"{H}/emb/B_lid3_voc.npy"); langs = json.load(open(f"{H}/emb/lid_langs.json"))
CODES = {"en": "english", "zh": "chinese", "nl": "dutch", "fr": "french", "de": "german", "it": "italian", "ja": "japanese", "pl": "polish", "pt": "portuguese", "es": "spanish"}; src_lang = {r["sample_id"]: CODES.get(langs[int(np.argmax(lid[i]))], "english") for i, r in enumerate(rows)} if lid.shape[1] == len(langs) else None
if src_lang is None: src_lang = {r["sample_id"]: "english" for r in rows}; print("language-id width mismatch: all en", lid.shape, len(langs))
os.makedirs(a.outdir, exist_ok=True); tmp = f"{a.outdir}/_tmp"; os.makedirs(tmp, exist_ok=True)
def vocal_wav(sid):
    p = f"{tmp}/{sid}_voc.wav"
    if not os.path.exists(p): z = np.load(f"{H}/stems/B/{sid}.npz"); wavfile.write(p, 24000, (np.clip(z["vocals"].astype(np.float32), -1, 1) * 32767).astype(np.int16))
    return p
hps = utils.HParams(**utils.get_hparams_from_file(a.hpfile)); enc = ECAPA2SpeakerEncoder16k(device="cuda")
pitch_predictor = get_f0_predictor("rmvpe", sampling_rate=hps.data.sampling_rate, hop_length=hps.data.hop_length, device="cpu", threshold=0.05)
net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model, config=hps).cuda().eval(); utils.load_checkpoint("hf_weights/G_00014_0225000.pth", net_g, None, True); lang2id = hps.data.lang2id
G = {}
with torch.no_grad():
    for m in MK:
        ids = sorted(r["sample_id"] for r in rows if r["split"] == "train" and r["label"] == m)[:a.K]; E = torch.stack([enc.get_speaker_embedding(vocal_wav(s)).cuda().reshape(-1) for s in ids]); G[m] = E.mean(0, keepdim=True); print("market voice", m, len(ids), flush=True)
val = sorted(r["sample_id"] for r in rows if r["split"] == "validation"); n = 0
TREF = {}
if a.nearest:   # voice-matched reference: for every market the TRAIN vocal stem whose ECAPA2 embedding is closest (cosine) to the source clip's embedding
    with torch.no_grad():
        for m in MK: TREF[m] = torch.stack([enc.get_speaker_embedding(vocal_wav(r["sample_id"])).cuda().reshape(-1) for r in rows if r["split"] == "train" and r["label"] == m]); print("train voices", m, TREF[m].shape[0], flush=True)
with torch.no_grad():
    for sid in val:
        wav, _ = librosa.load(vocal_wav(sid), sr=hps.data.sampling_rate); y = torch.from_numpy(wav)[None, None].cuda(); lid_src = torch.tensor(lang2id[src_lang[sid]]).unsqueeze(0).cuda()
        pitch = np.asarray(pitch_predictor.compute_f0(wav), dtype=np.float32).reshape(-1); pitch = np.clip(pitch, 0, 800) * a.pitch_factor; zl = round(y.shape[-1] / hps.data.hop_length)
        pitch = torch.nn.functional.interpolate(torch.tensor(pitch).unsqueeze(0).unsqueeze(0), size=zl, mode="nearest").squeeze().unsqueeze(0).unsqueeze(0).cuda()
        MKS = ["self"] if a.selfvoice else MK
        if a.nearest:
            e0 = enc.get_speaker_embedding(vocal_wav(sid)).cuda().reshape(1, -1)
            for m in MK: sim = torch.nn.functional.cosine_similarity(TREF[m], e0, dim=1); G[m] = TREF[m][int(sim.argmax())][None]
        if a.selfvoice: G["self"] = enc.get_speaker_embedding(vocal_wav(sid)).cuda().reshape(1, -1)
        for m in MKS:
            fp = f"{a.outdir}/{sid}_{m}.wav"
            if os.path.exists(fp): continue
            try: out = net_g.voice_conversion(c_src=None, y_src=y, y_tgt=None, g_tgt=G[m].unsqueeze(-1), mel_tgt=None, c_lengths=None, pitch_tgt=pitch, lang_id_src=lid_src)[0][0].data.cpu().float().numpy()
            except Exception as e:
                open(f"{a.outdir}/failures.txt", "a").write(f"{sid} {m} {src_lang[sid]} pitch {tuple(pitch.shape)} y {tuple(y.shape)} {str(e)[:120]}\n"); print("FAILED", sid, m, str(e)[:100], flush=True); continue
            wavfile.write(fp, hps.data.sampling_rate, (np.clip(out, -1, 1) * 32767).astype(np.int16)); n += 1; print(sid, m, src_lang[sid], n, flush=True)
        if a.limit and n >= a.limit * (1 if a.selfvoice else 6): break
print("done", n)
