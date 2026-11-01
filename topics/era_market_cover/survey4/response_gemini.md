Several systems released between 2024 and 2026 support melody-preserving cover generation via text and audio or symbolic conditioning.  
**JASCO**

* **Link & Status:** Runnable weights on Hugging Face (facebook/jasco-chords-drums-melody-\*).  
* **License & Requirements:** CC BY-NC 4.0 weights, MIT code. Input text/audio/MIDI; Output WAV. Requires PyTorch/AudioCraft environment.  
* **Conditioning:** Text prompt \+ chords, drums, and optionally melody.  
* **Duration & Era Prompt:** 10 seconds. Supports era/decade styles.

**Music ControlNet**

* **Link & Status:** Pretrained models at MusicControlNet.github.io.  
* **License & Requirements:** CC BY-NC-ND 4.0. Input text/audio; Output audio. *Inference hardware requirements unverified at source.*  
* **Conditioning:** Text \+ time-varying controls (melody, rhythm, dynamics).  
* **Duration & Era Prompt:** Supports era prompts. *Max duration unverified at source.*

**MuseControlLite**

* **Link & Status:** Checkpoints via MuseControlLite.github.io/web/.  
* **License & Requirements:** CC BY 4.0. Input text/audio; Output audio. *Inference hardware requirements unverified at source.*  
* **Conditioning:** Text \+ attributes \+ audio (melody, rhythm, dynamics).  
* **Duration & Era Prompt:** Supports era prompts. *Max duration unverified at source.*

**SongEcho**

* **Link & Status:** Models available via \[github.com/lsfhuihuiff/SongEcho\_ICLR2026\](https\://github.com/lsfhuihuiff/SongEcho\_ICLR2026).  
* **License & Requirements:** *License unverified at source.* Input text/audio; Output audio. Requires NVIDIA GPUs.  
* **Conditioning:** Text genre tags \+ vocal melody.  
* **Duration & Era Prompt:** 240 seconds. Supports era/genre tags.

**MPEcho**

* **Link & Status:** Checkpoints available at \[github.com/YatingMusic/MPEcho\](https\://github.com/YatingMusic/MPEcho).  
* **License & Requirements:** *License and inference hardware requirements unverified at source.* Input text/audio; Output audio.  
* **Conditioning:** Text \+ melody \+ phoneme-aware embeddings.  
* **Duration & Era Prompt:** Supports era prompts. *Max duration unverified at source.*

**MuLaCover**

* **Link & Status:** Weights on Hugging Face (HeartMuLa/MuLaCover).  
* **License & Requirements:** CC BY-NC 4.0 weights (non-commercial output), Apache 2.0 code. Input text \+ audio/MIDI; Output WAV. Requires GPU.  
* **Conditioning:** Text style description \+ reference audio OR melody/chord MIDI.  
* **Duration & Era Prompt:** Full-song length (*exact max duration limit unverified*). Supports era prompts.

**Stable Audio Open ControlNet**

* **Link & Status:** Checkpoints at stable-audio-control.github.io.  
* **License & Requirements:** Stability AI Non-Commercial Research License. Input text/audio; Output audio.  
* **Conditioning:** Text \+ input audio (CQT/Chroma representation).  
* **Duration & Era Prompt:** 47 seconds. Supports era prompts.