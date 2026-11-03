# **Exhaustive Analysis of Cross-Axis Controllable Music Generation: Time and Market Transfer**

The "Time Traveler" project seeks to establish a generative framework capable of rendering same-song covers controlled independently along two orthogonal axes: temporal stylistic evolution (era of release) and geographic/cultural localization (market). Preliminary probes establish a foundational hypothesis: temporal production signatures (decade information) are predominantly encoded within the instrumental accompaniment, whereas geographic market information relies heavily on linguistic markers, vocal timbre, and delivery. The current implementation, utilizing a frozen MusicGen-Melody backbone augmented with a learned era-axis adapter, successfully manipulates the accompaniment while preserving the original vocal and melodic structures. However, market control derived solely from the accompaniment remains structurally weak, exhibiting a 6-way classification accuracy of 0.22 to 0.33 against a chance baseline of 0.167.  
To overcome these limitations and expand the zero-budget, GPU-equipped academic effort in Yokohama, Japan, this report provides a rigorous, exhaustive analysis of available datasets, representation encoders, generative architectures, evaluation paradigms, and the specific legal frameworks governing text and data mining in Japan. Facts extracted from the literature are strictly isolated from the interpretative analysis synthesizing these components into a viable research pipeline.

## **1\. Original Discogs-VI and Discogs-VI-YT (Araz et al., ISMIR 2024\)**

The Discogs-VI and Discogs-VI-YT datasets, introduced by Araz, Serra, and Bogdanov at ISMIR 2024, represent massive-scale resources for musical version identification (VI) and cover song identification (CSI)1. Because commercial audio distribution violates copyright law, the authors constructed a relational metadata architecture linking public editorial tags to third-party streaming identifiers.

| Option | URL / DOI | Size | Labels & Metadata | Access Route | License |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Discogs-VI (Full)** | 10.5281/zenodo.13983028 | \~1.9M versions, 348K cliques | Clique ID, Artist, Writer, Title, Genre, Style, Label, Year, Country | Zenodo (main.zip) | Metadata: CC BY-NC-SA 4.0 |
| **Discogs-VI-YT** | 10.5281/zenodo.13983028 | 493K versions, 98K cliques | Clique ID, Version ID, Matched YouTube IDs (avg 1.4 per track) | Zenodo (main.zip) | Metadata: CC BY-NC-SA 4.0 |
| **CQT Audio Features** | github.com/raraz15/Discogs-VINet | 493K pre-computed arrays | Constant-Q Transform (CQT) arrays | Request via MTG authors | Non-commercial research |
| **Discogs-VINet Code** | github.com/raraz15/Discogs-VINet | 5.2M parameters | Trained network weights, embeddings | GitHub branch ISMIR2024 | Code: Affero GPLv3 |

### **Dataset Composition and Technical Specifications**

The primary artifact released is a suite of JSONL files mapping track versions into "cliques" (groups of tracks representing versions of the same underlying musical work) using metadata matching on artist credits, writer credits, and track titles1. The Discogs-VI-20240701.jsonl file contains the raw, unmapped metadata (7 GB uncompressed), while the Discogs-VI-YT-20240701.jsonl subset provides YouTube ID linkages1. The authors also provide a lightweight split (Discogs-VI-YT-light-20240701.json) partitioned into train, validation, and test sets that explicitly exclude intersections with the Da-TACOS and SHS100K test sets to prevent evaluation contamination1.  
The audio features utilized to train the baseline model (Discogs-VINet) are pre-computed Constant-Q Transform (CQT) representations1. The Discogs-VINet architecture—an adaptation of CQT-Net containing 5.2 million parameters—projects these CQT features into a 512-dimensional linear space, appending an L2 normalization layer to ensure the generated version embeddings map to a unit hypersphere4. The network is trained using a Triplet Loss function, bypassing classification heads due to the extreme number of cliques (98,000 classes), utilizing mini-batches of 48 distinct cliques with 2 versions per clique4. While the linear projection is confirmed at 512 dimensions, the precise bin count, hop length, and frequency bounds of the raw CQT inputs released to researchers remain UNVERIFIED within the immediate Zenodo repository documentation, though they follow the default configurations of the original CQT-Net stored in 16-bit precision1.

### **Recommended Route for a Three-Person GPU-Equipped Team**

For a team operating with a GPU server and no external budget, relying purely on pre-computed features restricts the ability to train an autoregressive or diffusion-based audio generator. The required route is to download the Discogs-VI-YT JSONL files and utilize the provided Python script download\_missing\_version\_youtube\_urls.py to scrape the raw waveforms1. The team should parallelize the download process using the provided utilities/shuffle\_and\_split.sh script to shard the JSONL file1.

### **Open Risks and Caveats**

The most significant operational bottleneck is YouTube's automated rate-limiting architecture. The dataset authors explicitly warn that executing more than 10 parallel download processes will trigger a sustained IP ban from YouTube, necessitating a multi-day cessation of all network requests4. Furthermore, because individual YouTube videos are routinely deleted, geoblocked, or privatized, download attrition is inevitable. Researchers must execute discogs\_vi\_yt/post\_processing.py to drop any cliques where fewer than two versions were successfully retrieved, which shrinks the usable dataset continuously over time1. Legally, the authors conducted their scrapes under the European Union's Directive 2019/790 text and data mining (TDM) exception (Article 3\)4. The legality of this scraping mechanism under the jurisdiction of the Japanese Copyright Act is addressed extensively in Section 7\.

## **2\. Alternative Multi-Axis Music Datasets**

To train a robust two-axis generative model capable of manipulating time and market independently, the training corpus must contain dense, accurate labels for both the release year (era) and the geographic market (country), seamlessly paired with retrievable audio.

| Dataset | URL / DOI | Size | Labels (Year, Country, Audio) | Access Route | License / Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Da-TACOS** | 10.5281/zenodo.5528870 | 15,000 versions, 1,000 cliques | SHS metadata, Release Year, Pre-extracted audio features | Zenodo / mirdata library | Features: CC BY 4.0 |
| **SHS100K** | github.com/NovaFrost/SHS100K | \~116K tracks, 9,359 cliques | SHS metadata, Audio via YouTube links | Third-party YT scraping | Metadata: CC BY-NC-SA 4.0 |
| **DALI** | github.com/gabolsgabs/DALI | 5,358 (v1) / 7,756 (v2) tracks | Lyrics, Vocal notes, Release Year, Country, YT audio | Request via authors / Zenodo | Metadata provided, audio fetched |
| **Music4All-Onion** | Moscati et al., 2022 | 109,269 tracks | Year, Country, Spotify tags, 30s center-cropped audio | Request via institution | Academic use only6 |
| **MusicBrainz Canonical** | metabrainz.org | \~38M rows | Canonical release, Year, Country | Direct PostgreSQL dump | Public Domain (CC0) |

### **Dataset Analysis and Interpretation**

**Da-TACOS and SHS100K:** Both datasets are constructed upon the SecondHandSongs (SHS) API, a community-driven database of cover songs3. Da-TACOS provides a highly curated benchmark subset of 15,000 versions, but crucially, it distributes pre-extracted audio features rather than raw audio or URLs9. This renders it incompatible with text-to-audio generative frameworks that require waveform reconstruction. SHS100K scales up to 116,352 tracks and provides YouTube links8. However, recent replication studies note severe link rot within SHS100K; researchers attempting to scrape the test subset (SHS-SEED) found that only \~82% of the dataset is currently retrievable7.  
**DALI (Dataset of synchronised Audio, LyrIcs and notes):** DALI is uniquely positioned for the market-axis manipulation objective. Constructed using a teacher-student machine learning paradigm, it provides highly granular, time-aligned vocal melody notes and lyrics at four levels of granularity: syllable, word, line, and segment12. The dataset contains 5,358 tracks in version 1 and 7,756 tracks in version 213. Crucially, the metadata includes the release year and the geographic country of the singer/band, allowing researchers to isolate vocal phonemes and stylistic markers tied directly to regional markets13.  
**Music4All and Music4All-Onion:** This multimodal dataset features 109,269 tracks represented by 30-second center-cropped audio clips, aligned with lyrics, Last.fm semantic tags, and Spotify popularity scores6. The tracks are strictly filtered to remove pre-1960 recordings6. The statistical distribution of release years is heavily skewed toward recent decades, possessing a mean year of 2006 and a median of 20116. The Onion extension appends 253 million user listening events for 56,512 of these tracks, offering dense user-item interaction profiles6.  
**MusicBrainz Canonical Metadata:** Released in 2023, this specialized dataset aims to solve the proliferation of duplicated recordings across global markets. It maps millions of duplicated audio recordings to a single "canonical" release, determined programmatically by ordering all releases by format (preferring digital over analog), earliest release date, and release country22.

### **Recommended Route for a Three-Person GPU-Equipped Team**

The optimal path to isolate and manipulate the market axis (which probes confirm is deeply tied to vocals) is to fuse the **DALI dataset** with **MusicBrainz Canonical Metadata**. DALI provides the exact time-aligned vocal sequences, lyrics, and corresponding YouTube audio links13. The team can query the MusicBrainz PostgreSQL dump using the DALI track metadata to retrieve the canonical release\_group entity, ensuring the appended release year and geographic country are accurate to the original composition rather than a subsequent compilation album22.

### **Open Risks and Caveats**

The primary risk across all discography datasets is metadata hallucination induced by reissue confusion. Re-releases and digital remasters frequently inherit the copyright date of the original track or the date the remaster was uploaded to a streaming server, completely corrupting the temporal axis25. Geographic tags are similarly fragile; a track may be labeled by the country of a secondary market release (e.g., an "Import" date on Amazon or Discogs) rather than the artist's true geographic origin25. This necessitates an acoustic filtering step to verify that the audio waveform matches the expected temporal artifacts.

## **3\. Pair Construction: Filtering Reissues and Remasters**

Building trustworthy training pairs of different recordings of the same composition requires distinguishing true stylistic covers from re-releases, digital remasters, or minor radio edits. If an era-adapter is trained on a "cover" that is merely a 2015 high-definition remaster of a 1980s original track, the generative model will learn to associate the "2010s" era with digital mastering artifacts (e.g., EQ curves, multi-band compression, loudness war clipping) rather than true stylistic, rhythmic, or harmonic evolution.

| Strategy | Implementation Tool | Description | Access |
| :---- | :---- | :---- | :---- |
| **Relational Metadata** | MusicBrainz Release Groups | Abstracts multiple physical releases into one canonical group | Public Database22 |
| **Cover Identification Models** | CoverHunter | Extracts 128-d embeddings to calculate structural acoustic distance | Open Source27 |
| **Acoustic Fingerprinting** | CLEWS (Discogs-VI-SIREN) | Locates identical audio regions to flag exact mastering duplicates | Open Source1 |

### **Methodological Approach to Exclusion**

The architectural foundation of pair construction begins in the metadata layer. MusicBrainz organizes audio through the release\_group entity, which abstracts multiple distinct physical releases (e.g., the 1984 vinyl, the 1992 CD reissue, and the 2015 digital remaster) into a single compositional bucket24. By querying the firstreleasedate via the MusicBrainz API, researchers can extract the absolute earliest origin era of the composition32.  
However, metadata is insufficient to guarantee acoustic divergence. The team must apply an acoustic proxy model to filter out identical performances masquerading as covers. CoverHunter (Liu et al., 2023\) is a state-of-the-art model designed for this exact thresholding task. It utilizes a convolution-augmented transformer (Conformer) backbone paired with a novel coarse-to-fine chunk alignment scheme9. CoverHunter processes 15-second short chunks to learn initial alignments, extending them to 45-second chunks for fine training28. Trained with a joint focal, center, and triplet loss function, the model extracts highly discriminative 128-dimensional embeddings, achieving a Mean Average Precision (MAP) of 0.86 on the SHS100K-Test dataset and 0.85 on Da-TACOS28.  
Alternatively, the team can utilize the Discogs-VINet architecture, which provides 512-dimensional embeddings, or the CLEWS (SImilar REgioNs) framework introduced alongside Discogs-VI to actively search for identical localized audio regions, serving as a high-precision acoustic fingerprint to detect remasters1.

### **Recommended Route**

> 1. Query the MusicBrainz API for the release\_group to group compositions and extract the earliest release\_date and country22.  
> 2. Retrieve the audio waveforms from YouTube using the matched metadata.  
> 3. Pass all candidate pairs through the pre-trained CoverHunter model (utilizing the open-source 128-d compact gate implementation)27.  
> 4. Calculate the pairwise cosine similarity. Pairs that yield a similarity above an explicit threshold (e.g., \>0.95) must be discarded as remasters or minor edits. Pairs scoring between 0.60 and 0.90 should be retained as true stylistic covers exhibiting temporal or geographic variation4.

### **Open Risks and Caveats**

The interpretation of acoustic similarity presents a distinct risk regarding live performances. Live bootlegs25 or acoustic "unplugged" versions performed by the original artist will score significantly lower in similarity, potentially entering the dataset as valid "covers" from a different era. If the dataset absorbs too many live renditions, it biases the model's understanding of temporal style toward live acoustic performance metrics rather than standard studio production evolution.

## **4\. Era and Market Prediction from Audio (Probes)**

To iteratively train and evaluate the generative adapters, the research team requires highly accurate classifiers (probes) capable of predicting the release decade and the geographic market directly from the audio waveform, independent of metadata.

| Prediction Task | Best Available System / Encoder | Reported Performance | Known Confounds |
| :---- | :---- | :---- | :---- |
| **Market (Geography)** | Late-fusion Deep Learning (MFCC \+ Language) | 33.03% (56-way class), MGCE: 2,754 km | Regional blending, extreme class imbalance36 |
| **Era / Metadata** | MuQ (Mel-RVQ) / MuQ-MuLan | SOTA zero-shot tagging (79.3 ROC-AUC) | Audio degradation, remastering bias38 |
| **General Audio** | Audio Spectrogram Transformer (AST) | 0.485 mAP (AudioSet), 95.6% (ESC-50) | High data requirements (ImageNet pretraining)41 |

### **Market and Geographic Prediction**

Predicting the geographic origin of music strictly from audio is a highly complex, non-linear classification problem. A recent study by Zhou et al. evaluating a dataset of 1,142 recordings across 73 countries utilized K-Nearest Neighbors (KNN) and Random Forest regression models built on MARSYAS features, achieving a Mean Great-Circle Error (MGCE) of 3,113 km36. An updated late-fusion deep learning architecture processing Mel-Frequency Cepstral Coefficients (MFCCs) simultaneously with language features improved these metrics, achieving an accuracy of 33.03% across 56 countries (vastly outperforming the random baseline of 1.79%) and reducing the MGCE to 2,754.83 kilometers36.  
The interpretation of this data suggests that market localization from pure audio is highly prone to geographic blending; bordering nations frequently share indistinguishable harmonic scales and rhythmic traditions36. Furthermore, the datasets exhibit extreme class imbalance, with Western Pop, Indian classical, and specific African regional music heavily overrepresented relative to smaller sovereign states37.

### **Era and Representation Encoders**

For era prediction and structural acoustic modeling, the Audio Spectrogram Transformer (AST) introduced by Gong et al. represents a convolution-free, purely attention-based architecture41. AST converts a 10-second audio waveform into a 128x100t log Mel filterbank sequence, splits the spectrogram into a sequence of 16x16 overlapping patches, flattens them into 768-dimensional embeddings, and processes them through a standard Transformer encoder41. Transferred from ImageNet pretraining, AST achieves 0.485 mAP on AudioSet and 95.6% accuracy on ESC-5041.  
Alternatively, the newly published MuQ (Self-Supervised Music Representation Learning with Mel Residual Vector Quantization) provides state-of-the-art tokenized target prediction38. Unlike MERT—which relies on the computationally heavy EnCodec model and auxiliary CQT losses—MuQ utilizes a single-layer Mel-RVQ tokenizer to generate discrete targets38. Trained via Masked Language Modeling (MLM) with a masking probability of 0.6, MuQ incorporates distinct linear layers to predict different target tokens across the Mel spectrum38. Utilizing iterative training from 0.9K hours (Music4All) up to 160K hours of audio, MuQ captures both low-level acoustic properties and high-level semantic concepts, achieving a 79.3 ROC-AUC score on the MagnaTagATune zero-shot music tagging task when paired with the MuLan text-contrastive model (MuQ-MuLan)38.

### **Recommended Route**

Utilize **MuQ-MuLan** as the foundational backbone encoder for all probe classifiers. Because MuQ utilizes masked language modeling against highly stable Mel-RVQ tokens, it demands significantly less annotated data for fine-tuning; recent evaluations demonstrate that LoRA-adapted MuQ models achieve high Spearman rank correlations with as few as 150 training clips39. The team should freeze the MuQ embeddings and train a lightweight era regressor and a geographic multi-class probe on top of them46.

### **Open Risks and Caveats**

The primary confounds include dataset bias (an overrepresentation of US/UK pop music dictates the underlying latent space) and linguistic leakage37. The market probe will likely overfit to the language of the vocals rather than the underlying musical style. If the team attempts to transfer a US song to a Spanish market, the probe might simply search for Spanish phonemes, ignoring the rich stylistic nuances of flamenco or Latin pop production.

## **5\. Controllable Cover and Style Transfer Generation**

Generative music models have advanced beyond unconditional text-to-audio generation, enabling multi-axis structural and stylistic control. To fulfill the "Time Traveler" objective, the system must retain the core melody and lyrical content while fundamentally altering the production style, instrumentation, era, and regional vocal timbre.

| Generative System | Conditioning Mechanism | Primary Function / Axes | Evaluation Metrics |
| :---- | :---- | :---- | :---- |
| **MusicGen-Melody** | 12-pitch chromagram | Melody retention \+ Text control | FAD, KL Divergence, Chromagram accuracy47 |
| **MusiConGen** | Pre-pended Chords \+ Rhythm curves | Harmonic and Rhythmic structural control | FAD, majmin/triad correctness48 |
| **MusicLDM** | Text \+ Beat-synchronous mixup | High-fidelity text-to-music generation | FAD, Subjective listening tests50 |
| **So-VITS-SVC / Diff-SVC** | Pitch contour \+ Speaker embeddings | Singing Voice Conversion (Timbre/Market) | NMOS, Similarity Scores54 |

### **System Analysis for Instrumental Generation**

**MusicGen-Melody:** This variant (available in 1.5B and 3.3B parameter sizes) operates by extracting the pitch contour from a reference audio clip using a 12-pitch class chromagram operating at a 50Hz frame rate47. The autoregressive transformer enforces the retention of the original melody, while the concatenated textual prompt alters the genre, era, or production style, outputting tokens to the EnCodec vocoder59.  
**MusiConGen:** An advanced extension of the MusicGen architecture that introduces explicit temporal conditioning for chords and rhythm48. MusiConGen processes time-varying symbolic chords (translated into a 12-pitch chromagram) and rhythm conditions (derived from beat and downbeat probability curves concatenated with energy volume curves)49. These frame-wise additive conditions are injected directly into the self-attention blocks of the transformer via a condition gate control49. This capability is critical for era-shifting; the strict, quantized rhythmic grid of 1980s synth-pop requires vastly different beat probability curves than the swung, syncopated rhythms of 2020s trap or 1970s funk.  
**MusicLDM:** Utilizing a latent diffusion model optimized via rectified flow training (up to 200K steps), MusicLDM incorporates beat-synchronous mixup strategies to generate high-fidelity, highly novel music from text descriptions50. While powerful for unconditional generation, diffusion models generally struggle with strict frame-wise melodic anchoring compared to the specialized MusicGen-Melody architecture.

### **System Analysis for Vocal Conversion**

Because market information is predominantly embedded in vocals and language, the team must address the vocal track independently. Singing Voice Conversion (SVC) aims to convert a source singer's voice into a target singer's voice while retaining the source lyrics and melody54. Mainstream SVC models include So-VITS-SVC (SoftVC VITS) and Diff-SVC, both of which demonstrate high Naturalness Mean Opinion Scores (NMOS)54. Advanced frameworks like VibE-SVC push this further by separating the F0 contour into low-frequency and high-frequency components using inverse Discrete Wavelet Transforms (iDWT), allowing the model to apply a constant scaling factor to the vibrato extent and rate at inference time65. This allows a generic vocal track to be styled with the heavy vibrato characteristic of certain regional folk markets (e.g., Japanese Enka or Spanish Flamenco).

### **Recommended Route**

To prevent the temporal adapter from interfering with the market adapter, the team must deploy a decoupled dual-pipeline architecture.

> 1. **Stem Separation:** Isolate the original vocals from the accompaniment using a state-of-the-art source separation model (e.g., HTDemucs).  
> 2. **Accompaniment Generation (Time Axis):** Pass the instrumental stem through MusiConGen (or MusicGen-Melody with a custom era-adapter)49. Condition the model using the original 12-pitch chromagram, the extracted beat probability curves, and a text prompt dictating the target era (e.g., "1980s analog synthesizer production").  
> 3. **Vocal Conversion (Market Axis):** Process the isolated vocals through a zero-shot Singing Voice Conversion model (such as a Diff-SVC derivative) conditioned on a vocal embedding highly representative of the target geographic market54. Apply VibE-SVC scaling if regional vibrato is required65.  
> 4. **Recombination:** Merge the generated accompaniment and converted vocal stems.

### **Open Risks and Caveats**

Failure modes in autoregressive models like MusicGen include severe error accumulation and divergence during long inferences (often referred to as mode collapse or repetitive looping)61. Furthermore, language conversion remains a hard failure point for SVC. While timbre, pitch, and accent can be successfully altered (e.g., forcing an English vocal to adopt a Spanish phonetic timbre), true cross-lingual lyric translation while maintaining rhythmic meter requires advanced phoneme-to-note alignment models that are currently unstable and prone to hallucination56.

## **6\. Evaluating a Two-Axis Move: Leakage and Circularity**

Validating that a generated clip has successfully moved along the intended axes (Time and Market) without destroying the core melodic composition requires a rigorous, multi-faceted evaluation framework.

### **The Pitfalls of Circular Evaluation**

A critical vulnerability threatening the validity of this research is *circular evaluation*. Circular evaluation occurs when the generative model and the evaluation probe share the same underlying data, representation embeddings, or training objectives69. For example, if the era-adapter is trained using MuQ embeddings to push the generation toward a specific decade, and the evaluation relies on a MuQ-based classifier to verify that decade, the evaluation is inherently circular. The probe is not measuring a true acoustic stylistic shift; it is merely measuring the presence of the exact latent artifacts the generative model was explicitly optimized to produce71.  
The literature on AI evaluation establishes that claim-aware completion judging suppresses dangerous-misreport recall and creates a circular evaluation path72. Withholding contextual data via aware/blind interventions can shift recall by massive margins (e.g., 43.4 percentage points in GUI-agent tracing)74. To combat this, the evaluation metrics must be strictly orthogonal to the training reward mechanisms, creating a replayable evidence-policy boundary74.

### **Objective Measurement Framework**

> 1. **Era and Market Shift:** Utilize the Fréchet Audio Distance (FAD) employing an entirely distinct audio encoder—such as VGGish (trained on AudioSet) or CLAP (Contrastive Language-Audio Pretraining)—to measure the distributional distance between the generated covers and a held-out reference set of authentic tracks from the target era/market48. A lower FAD indicates the generated audio distribution closely matches the authentic target distribution.  
> 2. **Audio Quality:** Calculate ViSQOL (Virtual Speech Quality Objective Listener) scores to evaluate the raw audio reconstruction fidelity and detect artifact degradation induced by the EnCodec bottleneck61.  
> 3. **Melody Retention:** To ensure the core composition has not been destroyed by the temporal shift, compute the Dynamic Time Warping (DTW) distance between the 12-pitch chromagram of the original track and the chromagram of the generated track48. Alternatively, evaluate the frame-wise chord correctness using the mir\_eval package (measuring majmin, triad, and tetrad accuracy)48.  
> 4. **Axis Leakage:** Check for unintended entanglement. If shifting a song to a 1980s style inadvertently changes the perceived geographic market, the axes are entangled. Run cross-axis probes to verify that the market classifier output remains stable during a time-axis shift, tracking the leakage rate75.

### **Recommended Route**

Employ a strict hold-out strategy. Use MERT representations to train the adapter, but evaluate the final output exclusively using AST or MuQ classifiers38. Supplement objective metrics with blind A/B human subjective testing (Naturalness Mean Opinion Score \- NMOS, and Similarity Mean Opinion Score \- SMOS) to validate the FAD results, specifically asking human listeners to identify the perceived decade and region of the generated clips55.

### **Open Risks and Caveats**

Human evaluation is highly subjective, non-reproducible at scale, and expensive68. Conversely, objective metrics like FAD are overly sensitive to mastering volume, multi-band compression, and background noise. A track might be successfully classified by a probe as "1980s" simply because the AI generated a lower-fidelity, hiss-heavy waveform, rather than accurately capturing 1980s musical arrangement, leading to a false-positive evaluation48.

## **7\. Legal and Ethical Considerations: Japan's Copyright Act**

Because the research is taking place physically in Yokohama, Kanagawa, Japan, the data scraping, model training, and publication operate strictly under the jurisdiction of the Japanese Copyright Act. The legal framework dictating the use of copyrighted audio (such as YouTube rips or commercial Spotify streaming) for machine learning differs drastically from the United States' "Fair Use" doctrine or the European Union's TDM directives.

### **Article 30-4: The Non-Enjoyment Exception**

The Japanese Copyright Act, specifically Article 30-4, explicitly permits the exploitation of copyrighted works (including downloading, copying, and analyzing audio from YouTube) without permission from the rights holder, provided the use is for "information analysis" (such as AI training and text/data mining)78. The fundamental, overriding requirement is that the use must be for a "non-enjoyment purpose." This means the researcher or the AI is not consuming the work to appreciate or enjoy the thoughts, emotions, or artistic expressions contained within it, but is merely extracting data patterns82.

### **The Proviso of Unreasonable Prejudice**

However, Article 30-4 contains a critical, highly debated proviso: the exception does not apply if the exploitation would "unreasonably prejudice the interests of the copyright owner in light of the nature and purpose of the copyrighted work and the circumstances of its use"85.  
The interpretation of this legal boundary suggests a nuanced risk profile for generative AI. If an AI is trained on a highly concentrated dataset of a specific artist's discography to intentionally generate exact stylistic replicas (e.g., generating a "new" Beatles song that competes with their catalog), this may be deemed as competing directly with the original creator, thus unreasonably prejudicing their interests80. Conversely, training on a massive, randomized dataset of general 1980s music to learn broad temporal styles is widely considered protected under Article 30-478. Furthermore, under Japanese law, a technical measure such as a robots.txt file or an EU-style metadata opt-out mechanism does not, by itself, have the legal effect of uniformly excluding Article 30-4 exceptions, unless it aims to protect a commercially sold database80.

### **Publishing Generated Material and Data**

The research team must carefully bifurcate what is kept internal versus what is published.

* **Code and Features:** Publishing the codebase, the trained model weights, and the extracted latent embeddings (like CQT arrays or MuQ tokens) is legally protected, as these do not constitute a playable musical work80.  
* **Raw Audio:** Redistributing the raw YouTube audio files or MP3s is direct copyright infringement. The dataset must be published purely as a list of YouTube IDs, Discogs IDs, or MusicBrainz IDs, forcing end-users to scrape the audio themselves1.  
* **Generated Clips:** Japan has no safe harbor that comprehensively exempts model providers from liability for copyright infringement caused by AI output80. Publishing generated covers is acceptable for academic demonstration on a project page, provided the generation does not output an overfitted, recognizable, or verbatim sample of the original copyrighted master recording78.

### **Recommended Route**

Host the project data purely as relational metadata and feature vectors on Zenodo, utilizing an open license (e.g., CC BY-NC-SA 4.0 for metadata, Affero GPLv3 for code)1. Execute all audio downloading and model training locally on the Yokohama servers to fully benefit from the Article 30-4 exception78. Include a prominent legal disclaimer in the GitHub repository stating that the dataset scripts are provided strictly for non-enjoyment research purposes under Japanese law.

### **Open Risks and Caveats**

The legal landscape is actively shifting. Rights-holders globally are suing AI developers for training on copyrighted works. While Japan is highly permissive regarding the *input* of training data, if the AI generates an output that is demonstrably similar to a copyrighted work and relies heavily on that specific work, the researcher is liable for infringement at the *output* stage80.

## **8\. Final Recommendations: The Top Three Promising Routes**

Based on the exhaustive analysis of datasets, generative capabilities, filtering mechanisms, and legal frameworks, the following are the three most promising data routes for building cross-decade and cross-country pairs, ranked by viability for a zero-budget, GPU-equipped academic team.

> 1. **The DALI \+ MusicBrainz Canonical Metadata Fusion (Highly Recommended)**  
   * **Why:** DALI provides the most rigorous, time-aligned vocal and lyric data essential for isolating the market-axis (which relies entirely on vocals and language)12. MusicBrainz provides the canonical era and country tags required to escape reissue contamination22.  
   * **First Concrete Step:** Query the MusicBrainz API to cross-reference the 7,756 DALI track\_ids against the MusicBrainz release\_group entities, extracting the absolute earliest firstreleasedate and country for each composition, discarding any tracks that return ambiguous or heavily fragmented dates.  
> 2. **The Discogs-VI-YT Pipeline with CoverHunter Filtering**  
   * **Why:** This pipeline provides the absolute largest volume of pre-grouped cover songs (493K versions), offering the massive scale required to train robust temporal embeddings1.  
   * **First Concrete Step:** Execute the download\_missing\_version\_youtube\_urls.py script utilizing exactly 8 parallel workers to acquire the audio (staying below the 10-worker limit to avoid YouTube IP bans)4. Immediately pass the downloaded audio through a pre-trained CoverHunter model to calculate cosine similarity, pruning acoustic duplicates (remasters) scoring above a 0.90 threshold to ensure only true stylistic covers remain28.  
> 3. **MusiConGen Accompaniment Generation \+ Zero-Shot SVC Dual Generation**  
   * **Why:** This is the most mathematically sound generative method for avoiding axis leakage. MusiConGen handles the temporal era shift via precise rhythmic and harmonic conditioning, while the SVC model independently handles the geographic market shift via vocal timbre replacement49.  
   * **First Concrete Step:** Extract the 12-pitch chromagram and isolate the vocal track from a 30-second test sample using HTDemucs. Pass the instrumental stem through MusiConGen conditioned on a distinct decade prompt (e.g., "1970s Disco") to verify structural stability and compute the DTW metric before attempting the complex cross-lingual vocal substitution.

#### **Works cited**

> 1. Discogs-VI: A musical version identification dataset based on public, [https\://zenodo.org/records/13983028](https://zenodo.org/records/13983028)  
> 2. 25th ISMIR 2024: San Francisco, CA, USA \- DBLP, [https\://dblp.org/db/conf/ismir/ismir2024](https://dblp.org/db/conf/ismir/ismir2024)  
> 3. arXiv:2410.17400v1 \[cs.SD\] 22 Oct 2024, [https\://arxiv.org/pdf/2410.17400](https://arxiv.org/pdf/2410.17400)  
> 4. Discogs-VI dataset and code \- GitHub, [https\://github.com/MTG/discogs-vi-dataset](https://github.com/MTG/discogs-vi-dataset)  
> 5. Discogs-VI Dataset, [https\://mtg.github.io/discogs-vi-dataset/](https://mtg.github.io/discogs-vi-dataset/)  
> 6. Who Will Top the Charts? Multimodal Music Popularity Prediction via, [https\://arxiv.org/pdf/2512.06259](https://arxiv.org/pdf/2512.06259)  
> 7. (PDF) On the robustness of cover version identification models, [https\://www\.researchgate.net/publication/389742870\_On\_the\_robustness\_of\_cover\_version\_identification\_models\_a\_study\_using\_cover\_versions\_from\_YouTube](https://www.researchgate.net/publication/389742870_On_the_robustness_of_cover_version_identification_models_a_study_using_cover_versions_from_YouTube)  
> 8. AN ENHANCED DATASET FOR COVER SONG ANALYSIS \- IADIS, [https\://www\.iadisportal.org/ijwi/papers/2024220206.pdf](https://www.iadisportal.org/ijwi/papers/2024220206.pdf)  
> 9. On the Robustness of Cover Version Identification Models \- arXiv, [https\://arxiv.org/html/2501.01333v1](https://arxiv.org/html/2501.01333v1)  
> 10. A SEMI-SUPERVISED DEEP LEARNING APPROACH TO DATASET, [https\://archives.ismir.net/ismir2023/paper/000077.pdf](https://archives.ismir.net/ismir2023/paper/000077.pdf)  
> 11. Supervised Contrastive Learning from Weakly-Labeled Audio ... \- arXiv, [https\://arxiv.org/html/2502.16936v1](https://arxiv.org/html/2502.16936v1)  
> 12. Program \- MTG \- Music Technology Group \- UPF, [https\://www\.upf.edu/web/mtg/musictechnologyforum/program](https://www.upf.edu/web/mtg/musictechnologyforum/program)  
> 13. Lyrics segmentation via bimodal text–audio representation, [https\://www\.cambridge.org/core/journals/natural-language-engineering/article/lyrics-segmentation-via-bimodal-textaudio-representation/E54E76EE0348B48A0951365ACE66D116](https://www.cambridge.org/core/journals/natural-language-engineering/article/lyrics-segmentation-via-bimodal-textaudio-representation/E54E76EE0348B48A0951365ACE66D116)  
> 14. Creating DALI, a Large Dataset of Synchronized Audio, Lyrics, and, [https\://pdfs.semanticscholar.org/7396/51896fe0723a0fcf99207b08299fdcb20896.pdf](https://pdfs.semanticscholar.org/7396/51896fe0723a0fcf99207b08299fdcb20896.pdf)  
> 15. Contrastive Lyrics Alignment with a Timestamp-Informed Loss, [https\://openreview.net/pdf?id=peM113bm8Z](https://openreview.net/pdf?id=peM113bm8Z)  
> 16. Contrastive Lyrics Alignment with a Timestamp-Informed Loss, [https\://www\.research-collection.ethz.ch/server/api/core/bitstreams/1569974e-3b0f-46cf-b622-92411f1f7e26/content](https://www.research-collection.ethz.ch/server/api/core/bitstreams/1569974e-3b0f-46cf-b622-92411f1f7e26/content)  
> 17. Deep Learning Approaches for Automatic Sung Speech Recognition, [https\://etheses.whiterose.ac.uk/id/eprint/30716/1/Thesis%20Gerardo%20Roa%20Dabike.pdf](https://etheses.whiterose.ac.uk/id/eprint/30716/1/Thesis%20Gerardo%20Roa%20Dabike.pdf)  
> 18. Automatic Lyrics Transcription in Polyphonic Music \- ResearchGate, [https\://www\.researchgate.net/publication/335989851\_Automatic\_Lyrics\_Transcription\_in\_Polyphonic\_Music\_Does\_Background\_Music\_Help](https://www.researchgate.net/publication/335989851_Automatic_Lyrics_Transcription_in_Polyphonic_Music_Does_Background_Music_Help)  
> 19. Revisiting Content-Based Music Recommendation: Efficient Feature, [https\://bytez.com/docs/arxiv/2604.20847/paper](https://bytez.com/docs/arxiv/2604.20847/paper)  
> 20. Literature Survey: MultiModal Prediction for Music Popularity, [https\://www\.cfilt.iitb.ac.in/resources/surveys/2025/survey-yashchoudhary-multimodal-prediction-for-music-popularity.pdf](https://www.cfilt.iitb.ac.in/resources/surveys/2025/survey-yashchoudhary-multimodal-prediction-for-music-popularity.pdf)  
> 21. Who Will Top the Charts? Multimodal Music Popularity Prediction via, [https\://arxiv.org/html/2512.06259v1](https://arxiv.org/html/2512.06259v1)  
> 22. New dataset: MusicBrainz Canonical Metadata \- MetaBrainz Blog, [https\://blog.metabrainz.org/2023/06/12/new-dataset-musicbrainz-canonical-metadata/](https://blog.metabrainz.org/2023/06/12/new-dataset-musicbrainz-canonical-metadata/)  
> 23. A collection of music APIs, databases, and related tools \- GitHub Gist, [https\://gist.github.com/0xdevalias/eba698730024674ecae7f43f4c650096?permalink\_comment\_id=5754655](https://gist.github.com/0xdevalias/eba698730024674ecae7f43f4c650096?permalink_comment_id=5754655)  
> 24. ListenBrainz Documentation, [https\://listenbrainz.readthedocs.io/\_/downloads/en/latest/pdf/](https://listenbrainz.readthedocs.io/_/downloads/en/latest/pdf/)  
> 25. Release / Date \- MusicBrainz, [https\://musicbrainz.org/doc/Release/Date](https://musicbrainz.org/doc/Release/Date)  
> 26. How to catalog a CD collection and choose the right release \- Medium, [https\://medium.com/@avenirer/how-to-catalog-a-cd-collection-and-choose-the-right-release-a424357e76d0](https://medium.com/@avenirer/how-to-catalog-a-cd-collection-and-choose-the-right-release-a424357e76d0)  
> 27. CoverHunter: Cover Song Identification with Refined Attention and, [https\://arxiv.org/abs/2306.09025](https://arxiv.org/abs/2306.09025)  
> 28. Cover Song Identification with Refined Attention and Alignments, [https\://arxiv.org/html/2306.09025v1](https://arxiv.org/html/2306.09025v1)  
> 29. Finaliters blackpaper — vision & protocol, [https\://www\.finaliters.com/blackpaper](https://www.finaliters.com/blackpaper)  
> 30. How to Add a Release \- MusicBrainz, [https\://musicbrainz.org/doc/How\_to\_Add\_a\_Release](https://musicbrainz.org/doc/How_to_Add_a_Release)  
> 31. MusicBrainz Database / Schema, [https\://musicbrainz.org/doc/MusicBrainz\_Database/Schema](https://musicbrainz.org/doc/MusicBrainz_Database/Schema)  
> 32. MusicBrainz API / Search / RecordingSearch, [https\://musicbrainz.org/doc/MusicBrainz\_API/Search/RecordingSearch](https://musicbrainz.org/doc/MusicBrainz_API/Search/RecordingSearch)  
> 33. MusicBrainz API / Search, [https\://musicbrainz.org/doc/MusicBrainz\_API/Search](https://musicbrainz.org/doc/MusicBrainz_API/Search)  
> 34. Cover Song Identification with Refined Attention and Alignments, [https\://www\.researchgate.net/publication/371605995\_CoverHunter\_Cover\_Song\_Identification\_with\_Refined\_Attention\_and\_Alignments](https://www.researchgate.net/publication/371605995_CoverHunter_Cover_Song_Identification_with_Refined_Attention_and_Alignments)  
> 35. Release \- MusicBrainz, [https\://musicbrainz.org/doc/Release](https://musicbrainz.org/doc/Release)  
> 36. Predicting Music Origin with Deep Learning \- CSUSB ScholarWorks, [https\://scholarworks.lib.csusb.edu/context/etd/article/3526/viewcontent/Predicting\_Music\_Origin\_20250718.pdf](https://scholarworks.lib.csusb.edu/context/etd/article/3526/viewcontent/Predicting_Music_Origin_20250718.pdf)  
> 37. Predicting Geographic Origin in Traditional Music Works, [https\://www\.iccs-meeting.org/archive/iccs2024/papers/148330163.pdf](https://www.iccs-meeting.org/archive/iccs2024/papers/148330163.pdf)  
> 38. MuQ: Self-Supervised Music Representation Learning with Mel, [https\://www\.researchgate.net/publication/387671171\_MuQ\_Self-Supervised\_Music\_Representation\_Learning\_with\_Mel\_Residual\_Vector\_Quantization](https://www.researchgate.net/publication/387671171_MuQ_Self-Supervised_Music_Representation_Learning_with_Mel_Residual_Vector_Quantization)  
> 39. arXiv:2501.01108v2 \[cs.SD\] 3 Jan 2025, [https\://arxiv.org/pdf/2501.01108?](https://arxiv.org/pdf/2501.01108)  
> 40. MuQ: Self-Supervised Music Representation Learning with Mel, [https\://arxiv.org/html/2501.01108v1](https://arxiv.org/html/2501.01108v1)  
> 41. (PDF) AST: Audio Spectrogram Transformer \- ResearchGate, [https\://www\.researchgate.net/publication/354221509\_AST\_Audio\_Spectrogram\_Transformer](https://www.researchgate.net/publication/354221509_AST_Audio_Spectrogram_Transformer)  
> 42. (PDF) AST: Audio Spectrogram Transformer \- ResearchGate, [https\://www\.researchgate.net/publication/350647165\_AST\_Audio\_Spectrogram\_Transformer](https://www.researchgate.net/publication/350647165_AST_Audio_Spectrogram_Transformer)  
> 43. \[2104.01778\] AST: Audio Spectrogram Transformer \- arXiv, [https\://arxiv.org/abs/2104.01778](https://arxiv.org/abs/2104.01778)  
> 44. \[PDF\] AST: Audio Spectrogram Transformer \- Semantic Scholar, [https\://www\.semanticscholar.org/paper/AST%3A-Audio-Spectrogram-Transformer-Gong-Chung/0e2d8b8d81092037f9866c1ceddcebb87318e38b](https://www.semanticscholar.org/paper/AST%3A-Audio-Spectrogram-Transformer-Gong-Chung/0e2d8b8d81092037f9866c1ceddcebb87318e38b)  
> 45. MuQ: Self-Supervised Music Representation Learning with Mel, [https\://www\.alphaxiv.org/abs/2501.01108](https://www.alphaxiv.org/abs/2501.01108)  
> 46. MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music, [https\://arxiv.org/html/2603.22677v2](https://arxiv.org/html/2603.22677v2)  
> 47. A Tutorial on Generative Models for Music Audio \- arXiv, [https\://arxiv.org/html/2610.38252](https://arxiv.org/html/2610.38252)  
> 48. MusiConGen: Rhythm and Chord Control for Transformer-Based, [https\://arxiv.org/html/2407.15060v1](https://arxiv.org/html/2407.15060v1)  
> 49. (PDF) MusiConGen: Rhythm and Chord Control for Transformer, [https\://www\.researchgate.net/publication/382459472\_MusiConGen\_Rhythm\_and\_Chord\_Control\_for\_Transformer-Based\_Text-to-Music\_Generation](https://www.researchgate.net/publication/382459472_MusiConGen_Rhythm_and_Chord_Control_for_Transformer-Based_Text-to-Music_Generation)  
> 50. FLUX that Plays Music \- arXiv, [https\://arxiv.org/html/2409.00587v1](https://arxiv.org/html/2409.00587v1)  
> 51. MusicMagus: Zero-Shot Text-to-Music Editing via Diffusion Models, [https\://ijcai.org/proceedings/2024/864](https://ijcai.org/proceedings/2024/864)  
> 52. MusicLDM: Enhancing Novelty in Text-to-Music Generation ... \- arXiv, [https\://arxiv.org/abs/2308.01546](https://arxiv.org/abs/2308.01546)  
> 53. EditGen: Harnessing Cross Attention Control for Instruction-Based, [https\://arxiv.org/html/2507.11096v1](https://arxiv.org/html/2507.11096v1)  
> 54. DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion, [https\://www\.researchgate.net/publication/358457356\_DiffSVC\_A\_Diffusion\_Probabilistic\_Model\_for\_Singing\_Voice\_Conversion](https://www.researchgate.net/publication/358457356_DiffSVC_A_Diffusion_Probabilistic_Model_for_Singing_Voice_Conversion)  
> 55. LDM-SVC: Latent Diffusion Model Based Zero-Shot ... \- ISCA Archive, [https\://www\.isca-archive.org/interspeech\_2024/chen24e\_interspeech.pdf](https://www.isca-archive.org/interspeech_2024/chen24e_interspeech.pdf)  
> 56. arXiv:2306.14422v2 \[cs.SD\] 6 Jul 2023, [https\://arxiv.org/pdf/2306.14422](https://arxiv.org/pdf/2306.14422)  
> 57. Music AI in 2026: Suno, Udio, Stem Separation & Copyright \- MLAIA, [https\://www\.mlaia.com/blog-music-ai.html](https://www.mlaia.com/blog-music-ai.html)  
> 58. MusicGen-Stem: Multi-stem music generation and edition through, [https\://www\.researchgate.net/publication/390537008\_MusicGen-Stem\_Multi-stem\_music\_generation\_and\_edition\_through\_autoregressive\_modeling](https://www.researchgate.net/publication/390537008_MusicGen-Stem_Multi-stem_music_generation_and_edition_through_autoregressive_modeling)  
> 59. \[2306.05284\] Simple and Controllable Music Generation \- arXiv, [https\://arxiv.org/abs/2306.05284](https://arxiv.org/abs/2306.05284)  
> 60. A Tutorial on Generative Models for Music Audio \- arXiv, [https\://arxiv.org/pdf/2610.38252](https://arxiv.org/pdf/2610.38252)  
> 61. Continuous Audio Language Models \- arXiv, [https\://arxiv.org/html/2509.06926v1](https://arxiv.org/html/2509.06926v1)  
> 62. CoMoSVC: A Consistency Method for Singing Voice Conversion, [https\://www\.studocu.vn/vn/document/truong-dai-hoc-hung-vuong/cong-nghe-thong-tin/comosvc-a-consistency-method-for-singing-voice-conversion-2401/165655180](https://www.studocu.vn/vn/document/truong-dai-hoc-hung-vuong/cong-nghe-thong-tin/comosvc-a-consistency-method-for-singing-voice-conversion-2401/165655180)  
> 63. Leveraging Diverse Semantic-based Audio Pretrained Models for, [https\://arxiv.org/html/2310.11160v3](https://arxiv.org/html/2310.11160v3)  
> 64. svc-develop-team/so-vits-svc: SoftVC VITS Singing Voice Conversion, [https\://github.com/svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)  
> 65. Vibrato Expression Control for Singing Voice Conversion with, [https\://ieeexplore.ieee.org/iel8/10723155/10818373/11571023.pdf](https://ieeexplore.ieee.org/iel8/10723155/10818373/11571023.pdf)  
> 66. SaMoye: Zero-shot Singing Voice Conversion Model Based ... \- arXiv, [https\://arxiv.org/html/2407.07728v3](https://arxiv.org/html/2407.07728v3)  
> 67. Voice Conversion Challenge 2020 — Intra-lingual semi-parallel and, [https\://www\.isca-archive.org/vccbc\_2020/yi20\_vccbc.pdf](https://www.isca-archive.org/vccbc_2020/yi20_vccbc.pdf)  
> 68. The Singing Voice Conversion Challenge 2025: From Singer ... \- arXiv, [https\://arxiv.org/html/2509.15629v1](https://arxiv.org/html/2509.15629v1)  
> 69. An Application-Oriented Bottleneck-Driven Survey of World Models, [https\://www\.mdpi.com/2673-4117/7/10/561](https://www.mdpi.com/2673-4117/7/10/561)  
> 70. From descriptive to generative: foundation-model approaches for, [https\://academic.oup.com/bib/article/27/5/bbag554/8885502](https://academic.oup.com/bib/article/27/5/bbag554/8885502)  
> 71. Beyond the Leaderboard: A Survey of the Science of Evaluation, [https\://ieeexplore.ieee.org/iel8/6287639/6514899/11488866.pdf](https://ieeexplore.ieee.org/iel8/6287639/6514899/11488866.pdf)  
> 72. Who Verifies the Agent's Success Claim? Self-Report Leakage in, [https\://www\.researchgate.net/publication/416527704\_Who\_Verifies\_the\_Agent's\_Success\_Claim\_Self-Report\_Leakage\_in\_GUI-Agent\_Completion\_Judging](https://www.researchgate.net/publication/416527704_Who_Verifies_the_Agent's_Success_Claim_Self-Report_Leakage_in_GUI-Agent_Completion_Judging)  
> 73. Reliability-Guided Dynamic Fusion for Few-Shot Writer-Specific, [https\://www\.mdpi.com/1424-8220/26/19/6355/pdf](https://www.mdpi.com/1424-8220/26/19/6355/pdf)  
> 74. Who Verifies the Agent's Success Claim? Self-Report Leakage in, [https\://arxiv.org/html/2610.29609v1](https://arxiv.org/html/2610.29609v1)  
> 75. Role-Aware Graduated Disclosure for LLMs over Mixed-Sensitivity, [https\://arxiv.org/html/2610.28067v1](https://arxiv.org/html/2610.28067v1)  
> 76. Interpretable and Perceptually-Aligned Music Similarity with ... \- arXiv, [https\://arxiv.org/html/2601.19109v1](https://arxiv.org/html/2601.19109v1)  
> 77. FlowSonic: Stable Zero-Shot Music Editing via High-Order ... \- Bytez, [https\://bytez.com/docs/arxiv/2607.17526/paper](https://bytez.com/docs/arxiv/2607.17526/paper)  
> 78. WIPO Webcast, [https\://webcast.wipo.int/video/SCCR\_48\_2026-05-21\_PM\_88481f30-304d-4ef2-738e-08deaff9ce9f](https://webcast.wipo.int/video/SCCR_48_2026-05-21_PM_88481f30-304d-4ef2-738e-08deaff9ce9f)  
> 79. J-CHAT: Japanese Large-scale Spoken Dialogue Corpus for ... \- arXiv, [https\://arxiv.org/pdf/2407.15828](https://arxiv.org/pdf/2407.15828)  
> 80. AI & Intellectual Property 2026 \- Global Practice Guides \- Chambers, [https\://practiceguides.chambers.com/practice-guides/comparison/1664/20087/31886-31887-31888-31889-31890-31891-31892-31893-31894-31895](https://practiceguides.chambers.com/practice-guides/comparison/1664/20087/31886-31887-31888-31889-31890-31891-31892-31893-31894-31895)  
> 81. EU Considers Redesigning 'Opt-Out' for AI Training — What's at, [https\://note.com/ipconnet/n/n9b8631bf1c54?hl=en](https://note.com/ipconnet/n/n9b8631bf1c54?hl=en)  
> 82. The Flexible Copyright Exception for 'Non-Enjoyment' Purposes, [https\://www\.researchgate.net/publication/348560766\_The\_Flexible\_Copyright\_Exception\_for\_'Non-Enjoyment'\_Purposes\_-\_Recent\_Amendment\_in\_Japan\_and\_Its\_Implication](https://www.researchgate.net/publication/348560766_The_Flexible_Copyright_Exception_for_'Non-Enjoyment'_Purposes_-_Recent_Amendment_in_Japan_and_Its_Implication)  
> 83. AI Compliance FAQ \- MmowW, [https\://mmoww.net/ai/faq/](https://mmoww.net/ai/faq/)  
> 84. "Article 30-4 of the Copyright Act" \- The pain of people who have, [https\://note.com/hideya\_sugimoto/n/n10e393405850?hl=en](https://note.com/hideya_sugimoto/n/n10e393405850?hl=en)  
> 85. Generative AI: Copying or Creating? The Ninth Circuit Limits DMCA, [https\://www\.tmi.gr.jp/eyes/blog/2026/18962.html](https://www.tmi.gr.jp/eyes/blog/2026/18962.html)  
> 86. Japan (2025) \- Copyright Atlas, [https\://www\.copyrightatlas.com/details?type=country\&slug=jp](https://www.copyrightatlas.com/details?type=country&slug=jp)  
> 87. 【SAKURA Law Office \| Legal Update by Managing Partner, [https\://sakura-lawyers.jp/legal-update-03-en/](https://sakura-lawyers.jp/legal-update-03-en/)  
> 88. A Comprehensive Analysis of Copyright Issues in the Use of ... \- note, [https\://note.com/itlawyer/n/n34dbbc5746d7?hl=en](https://note.com/itlawyer/n/n34dbbc5746d7?hl=en)  
> 89. “General Understanding on AI and Copyright in Japan”, [https\://www\.bunka.go.jp/english/policy/copyright/pdf/94055801\_01.pdf](https://www.bunka.go.jp/english/policy/copyright/pdf/94055801_01.pdf)  
> 90. Translators as Invisible Teachers of AI Copyright, Translation ... \- arXiv, [https\://arxiv.org/html/2605.24842v1](https://arxiv.org/html/2605.24842v1)