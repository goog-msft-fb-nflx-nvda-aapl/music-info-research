"""T-4h-b step 2: SoulX-Singer annotation pipeline with the English ASR step replaced by Whisper large-v3 word timestamps (the shipped Parakeet/NeMo path needs a newer torch than the pinned 2.2). Everything else (vocal detection, F0, ROSVOT notes, phonemes) is the repository's own pipeline.
usage (from ext/SoulX-Singer, venv_sx): PYTHONPATH=. python 138_sx_ann.py <pipeline args as in example/preprocess.sh>"""
import sys, torch, numpy as np, soundfile as sf, librosa
import preprocess.tools.lyric_transcription as LT
from transformers import pipeline as hfpipe
ORIG = LT._ASREnModel
class WhisperEn:
    def __init__(self, model_path, device):
        self.p = hfpipe("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=device)
    def process(self, wav_fn):
        y, sr = sf.read(wav_fn, dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y; y = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        r = self.p({"raw": y, "sampling_rate": 16000}, return_timestamps="word", generate_kwargs={"language": "en"})
        words, ts = [], []
        for c in r["chunks"]:
            w = ORIG._clean_word(c["text"]); s, e = c["timestamp"]
            if w and s is not None: words.append(w); ts.append([float(s), float(e if e is not None else s + 0.3)])
        words, durs = LT._build_words_with_gaps(words, ts, wav_fn)
        import os
        f0 = os.path.splitext(wav_fn)[0] + "_f0.npy"
        if os.path.exists(f0): words, durs = LT._word_dur_post_process(words, durs, np.load(f0))
        return words, durs
LT._ASREnModel = WhisperEn
import runpy; sys.argv = ["preprocess.pipeline"] + sys.argv[1:]; runpy.run_module("preprocess.pipeline", run_name="__main__")
