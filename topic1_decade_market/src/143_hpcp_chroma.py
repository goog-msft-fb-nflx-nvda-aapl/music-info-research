"""P-3c: HPCP of the dataset-A clips with the acoss/Da-TACOS recipe (essentia std HPCP on whitened spectral peaks, 44.1 kHz, frame 4096, hop 512, 12 bins), converted to MusicGen-Melody's chroma format (235 frames per 30 s, one-hot per frame by argmax) -> emb/hpcp_chroma/<id>.npy. usage (from ~/era_market, venv_ess): python src/143_hpcp_chroma.py"""
import os, sys, numpy as np, essentia.standard as es
sys.path.insert(0, "src"); from common import ROOT, manifest
os.makedirs(f"{ROOT}/emb/hpcp_chroma", exist_ok=True); T = 235
for r in manifest("A"):
    fp = f"{ROOT}/emb/hpcp_chroma/{r['sample_id']}.npy"
    if os.path.exists(fp): continue
    y = es.MonoLoader(filename=r["path"], sampleRate=44100)(); fg = es.FrameGenerator(y, frameSize=4096, hopSize=512); w = es.Windowing(type="blackmanharris62"); sp = es.Spectrum()
    peaks = es.SpectralPeaks(magnitudeThreshold=0, maxFrequency=3500, minFrequency=100, maxPeaks=100, orderBy="frequency", sampleRate=44100); wh = es.SpectralWhitening(maxFrequency=3500, sampleRate=44100)
    hp = es.HPCP(sampleRate=44100, maxFrequency=3500, minFrequency=100, referenceFrequency=440, nonLinear=False, harmonics=8, size=12); H = []
    for f in fg: s = sp(w(f)); fr, m = peaks(s); H.append(hp(fr, wh(s, fr, m)))
    H = np.array(H); idx = np.array_split(np.arange(len(H)), T); C = np.stack([H[i].mean(0) for i in idx]); oh = np.zeros_like(C); oh[np.arange(T), C.argmax(1)] = 1.0; np.save(fp, oh.astype(np.float32))
print("done")
