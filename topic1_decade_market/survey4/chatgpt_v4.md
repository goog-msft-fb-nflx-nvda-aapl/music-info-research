# Time Traveler: Data, Pair Construction, Prediction, Generation, Evaluation, and Legal Research

This report distinguishes **reported facts** from **project interpretation**. I opened the primary dataset pages, repositories, proceedings pages, terms, and legal sources cited below. Where a requested property could not be established from a source I opened, it is marked **UNVERIFIED** rather than inferred.

## Discogs-VI and Discogs-VI-YT

### Verified release contents

| Option | URL | Size | What is released | Audio access | License / terms |
|---|---|---:|---|---|---|
| **Discogs-VI** — Recep Oğuz Araz, Xavier Serra, Dmitry Bogdanov, *ISMIR 2024* | [dataset](https://zenodo.org/records/13983028), [paper DOI](https://doi.org/10.5281/zenodo.14877379), [official site](https://mtg.github.io/discogs-vi-dataset/) | 1,911,611 versions; 348,796 works/cliques in the paper | Rich Discogs-derived version/work metadata; no YouTube requirement | No audio distributed | Metadata CC BY-NC-SA 4.0; code AGPLv3. citeturn153862search2turn393202search2turn785950view0 |
| **Discogs-VI-YT** — same authors, *ISMIR 2024* | [official site](https://mtg.github.io/discogs-vi-dataset/) | 493,049 versions; 98,785 cliques in the paper | Discogs-VI subset matched to official YouTube uploads; rich JSON plus a light clique/version/YouTube-ID file and train/val/test splits | IDs only; authors explicitly do **not** distribute audio | Same metadata license; downloaded-audio copyright is not granted by the dataset license. citeturn153862search1turn785950view0 |
| **Precomputed audio representations** | [official site](https://mtg.github.io/discogs-vi-dataset/) | Full-track representations for Discogs-VI-YT; exact storage size not stated on the opened page | Chroma, HPCP and CQT | Available **on request** for non-commercial scientific research | Research-only access condition stated by MTG. citeturn393202search0turn785950view0 |
| **Discogs-VINet** | [official code](https://github.com/raraz15/Discogs-VINet) | 5.2M parameters in ISMIR model; output embedding is 512-D | Model, weights, CQT extraction/training/inference code | Operates on locally available audio | Repository MIT license; contact listed as `recepoguz.araz@upf.edu`. citeturn785950view0turn785950view1 |

**Reported facts.** The downloadable Zenodo package contains `main.zip` at about 1.4 GB compressed / 21 GB uncompressed and an 8.8 GB intermediary archive. The rich `Discogs-VI-20240701.jsonl` and `Discogs-VI-YT-20240701.jsonl` are roughly 7 GB and 4 GB respectively; a light JSON contains clique IDs, version IDs and YouTube URLs. The authors additionally publish the larger `youtube_query_matched` mapping because availability varies geographically. citeturn153862search2turn153862search3

The Discogs-VI paper says the audio representations are **chroma, HPCP and CQT**. The baseline Discogs-VINet takes CQT computed using the CQT-Net setting and produces a **512-dimensional L2-normalized embedding**. I did **not** find a statement in the opened release documentation giving a fixed dimensionality for the distributed chroma/HPCP/CQT arrays themselves, so that requested number is **UNVERIFIED** rather than assumed. citeturn785950view0

The authors' reconstruction code downloads matched YouTube audio as mono, 44.1 kHz AAC; their crawl occupied about 1.8 TB before their model downsampled to 16 kHz on the fly. Their post-processing removes unavailable URLs, duplicate YouTube URLs within a clique, and cliques with fewer than two downloaded versions. citeturn153862search7turn393202search9

**Request procedure.** The public page says to contact the Music Technology Group for the precomputed representations, and the Discogs-VINet repository gives Araz's UPF email. I found no web form or published automatic approval process. Therefore the practical route is an academic email identifying the institution, PI/course, intended non-commercial use, requested feature subset, and storage constraints. citeturn393202search0turn785950view1

**YouTube downloads.** YouTube's current Terms of Service say users may not download or otherwise use Content except where the service expressly authorizes it or YouTube/rightsholders give permission, and they restrict automated access absent permission. YouTube API policy separately says API clients may not download, cache or store audiovisual content without prior written approval. citeturn589583search0turn589583search1 The Discogs-VI authors state that their institution downloaded under the EU Copyright in the Digital Single Market Directive's scientific text-and-data-mining exception; Article 3 covers reproductions/extractions by qualifying research organisations having lawful access and requires appropriate security. citeturn153862search2turn589583search4

**Interpretation.** The EU copyright exception and YouTube's contractual terms are separate questions. The Discogs-VI authors' legal basis does **not** amount to YouTube granting a research-download license. For a university outside that exact institutional/legal context, “the authors did it” should not be treated as sufficient authorization. citeturn589583search0turn589583search4

I found **no official audio distribution** for Discogs-VI-YT. I also found no community mirror that I could verify as an authorized redistribution of the commercial audio; community projects I found re-download from the supplied IDs instead. An authorized community audio copy is therefore **UNVERIFIED**. citeturn153862search2turn804359search17

**Recommended route for this team.** Use the released Discogs-VI metadata immediately and ask MTG for the precomputed features rather than beginning with a mass YouTube crawl. The metadata alone is unusually suitable for Time Traveler because its version cliques already solve much of the “same composition” problem and retain release-level Discogs information that can support year/country filtering. citeturn153862search1turn785950view0

**Open risks.** YouTube availability is time- and geography-dependent; the authors themselves warn that another crawl will not reproduce their exact downloaded set. Clique IDs were also documented as potentially changing when the dataset is regenerated from newer Discogs dumps. citeturn153862search1turn153862search2

## Datasets with year, market and usable audio information

The crucial distinction is between **release country** and **artist geographic origin**. Time Traveler needs the former if “US release rendered as Spain release” is the experimental target; artist nationality is not an equivalent label. MusicBrainz and Discogs explicitly model release country. citeturn268313search0turn268313search7

| Dataset | URL / citation | Size | Same-work grouping | Year + country? | Audio / features | License and main caveat |
|---|---|---:|---|---|---|---|
| **Discogs-VI / Discogs dumps** | [Araz et al., ISMIR 2024](https://doi.org/10.5281/zenodo.14877379) | 1.91M versions / 348.8k cliques in Discogs-VI | Yes, Discogs-VI cliques | Discogs release metadata contains date and country fields; Discogs-VI retains rich release metadata | YT IDs in VI-YT; representations by request | Best fit, but commercial audio is not redistributed. citeturn785950view0turn153862search1 |
| **MusicBrainz database** | [official database docs](https://musicbrainz.org/doc/MusicBrainz_Database) | Whole live MusicBrainz catalog; snapshot generated twice weekly | Recordings can link to a common **Work**; releases grouped into Release Groups | **Yes at release level:** date and country are core fields | No audio hosted | Core database CC0; work relationships are incomplete for some obscure material. citeturn268313search0turn268313search2turn268313search10 |
| **Da-TACOS** — Furkan Yesiler et al., *ISMIR 2019* | [dataset](https://mtg.github.io/da-tacos/), [paper](http://archives.ismir.net/ismir2019/paper/000038.pdf) | 15k benchmark tracks plus 10k cover-analysis tracks | Yes, SHS work IDs | Release **year yes**; release country not included in documented metadata | No audio; CENS, CREMA, HPCP, key, onset/tempo, MFCC and tags | Metadata/features CC BY-NC-SA 4.0; country must be enriched externally. citeturn726336view2turn804359search21 |
| **SHS-100K official 2025** — SecondHandSongs | [official repository](https://github.com/second-hand-songs/shs-100k/) | 10k works / 100k performances; separate validation/test additions described in repo | **Yes**, Work ID | Released CSV documents performance/work/title/artist/YouTube ID, **not year or country** | YouTube IDs | Excellent grouping; year/country require SHS API or external join and are therefore not available directly in released CSV. citeturn726336view1 |
| **SecondHandSongs API** | [API access](https://secondhandsongs.com/page/API/access) | Live database | Yes: original + covers are first-class relations | Performance metadata contains detailed recording/release information, but I could not verify a stable release-country field in the opened schema: **country field UNVERIFIED** | External links can include YouTube/Spotify/Apple Music | API key required for richer external-recording information. citeturn804359search2turn804359search12 |
| **MusicBrainz + AcousticBrainz** | [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_Database), [AcousticBrainz](https://acousticbrainz.org/download) | Large historical feature corpus keyed by MusicBrainz IDs | Via MB Work relations | MB supplies release year/country | Acoustic features, not commercial audio | AcousticBrainz data is CC0; project no longer needs source audio to use the dumps. citeturn268313search13turn268313search15 |
| **Million Song Dataset + SHS subset** | [MSD](https://millionsongdataset.com/) | 1M tracks; YearPredictionMSD 515,345 examples | SHS intersection supplies cover relationships | Release year available; release-country label is not a standard MSD field | Echo Nest-derived features, not raw audio | Useful historical benchmark, weak for your market axis. citeturn268313search12turn995251search0 |
| **DALI** — Gabriel Meseguer-Brocal, Alice Cohen-Hadria, Geoffroy Peeters, *ISMIR 2018; TISMIR 2020* | [dataset](https://zenodo.org/records/2577915), [TISMIR DOI](https://doi.org/10.5334/TISMIR.30) | 5,358 tracks in the original paper | Not designed as cover cliques | Release date and language are documented; release country **UNVERIFIED** | Synchronized lyrics, notes and associated audio references | Strong for vocal/language evaluation, weak for pair construction. citeturn804359search23turn804359search38turn804359search39 |
| **FMA** — Michaël Defferrard et al., *ISMIR 2017* | [dataset](https://archive.ics.uci.edu/dataset/386/fma+a+dataset+for+music+analysis) | 106,574 tracks / 917 GiB audio | No canonical same-work grouping | Release timestamps exist; release country is **not documented as a core track field** | Full MP3 audio + features | Audio is Creative-Commons material, but individual track licences matter; excellent legally accessible control corpus, poor direct market-pair source. citeturn729800search5turn729800search47 |
| **MTG-Jamendo** — Dmitry Bogdanov et al., *ICML workshop/MTG release 2019* | [official repository](https://github.com/MTG/mtg-jamendo-dataset) | 55k+ full tracks; `raw.meta.tsv` has 56,693 records | No same-work grouping | Release date yes; release country **UNVERIFIED / not documented in released metadata** | Full tracks from Jamendo | Metadata CC BY-NC-SA 4.0; dataset restricted by maintainers to non-commercial research/academic use; individual audio CC licenses apply. citeturn729800search0turn729800search11 |
| **Music4All A+A** — Jonas Geiger, Marta Moscati, Shah Nawaz, Markus Schedl, *CBMI 2025* | [paper/data](https://hcai.at/publications/2025_09_cbmi_music4all/) | 6,741 artists / 19,511 albums, linked to Music4All-Onion | Not a cover/version dataset | I found no verified per-recording release-country field in the opened source: **UNVERIFIED** | Multimodal data through parent dataset | CC BY-NC-SA 4.0; not a first-choice route for Time Traveler pairs. citeturn726336view3 |

**Reported fact on reissues.** MusicBrainz separates a **Recording** (the audio/performance entity), **Work** (composition), **Release** (a particular issue with country/date) and **Release Group** (the conceptual album/single). A single recording can therefore appear on multiple releases with different dates/countries. citeturn268313search0turn268313search4turn268313search7 This is extremely important: release-country metadata does not necessarily mean that a distinct recording was made for that market.

**Interpretation.** For Time Traveler, a MusicBrainz-only query should never create a “cross-country pair” merely because the same recording MBID appears on a US release and a Spanish release. That would be a distribution difference, not a cover-generation target. Use distinct recording IDs linked to the same Work, then attach the earliest defensible release metadata for each recording. citeturn268313search6turn268313search9

**Recommended route for this team.** Treat **Discogs-VI as the pairing backbone**, and enrich/validate with MusicBrainz rather than replacing Discogs-VI with SHS100K or Da-TACOS. Da-TACOS is valuable as a validation set because it gives explicit release years and trustworthy SHS work IDs, but it does not solve the country axis by itself. citeturn726336view2turn785950view0

**Open risks.** The largest source of label noise is confusing the date/country of a *physical or digital release* with the date/place characterizing the *recording*. Neither Discogs nor MusicBrainz automatically guarantees that those are the same concept. citeturn629722search3turn268313search7

## Constructing trustworthy cross-era and cross-market pairs

| Method | Source | What it gives | Use in Time Traveler |
|---|---|---|---|
| **Discogs-VI clique** | [Araz et al.](https://doi.org/10.5281/zenodo.14877379) | High-precision groups inferred from title, artist/writer credits and Discogs editorial metadata | Primary same-composition grouping. citeturn785950view0 |
| **SecondHandSongs Work ID** | [SHS-100K](https://github.com/second-hand-songs/shs-100k/) | Explicit original/cover work grouping | Strong external confirmation for overlapping material. citeturn726336view1turn729800search26 |
| **MusicBrainz Work → Recording** | [MusicBrainz Work docs](https://musicbrainz.org/doc/Work) | Composition entity linked to distinct recordings | Useful second independent graph for clique validation. citeturn268313search9 |
| **Discogs Master/Release metadata** | [Master Release guidelines](https://support.discogs.com/hc/en-us/articles/360005055493-Database-Guidelines-16-Master-Release) | Groups editions/pressings of the same release product | Primarily a **reissue filter**, not a same-song-cover label. citeturn629722search0turn629722search11 |
| **Discogs-VINet** | [code](https://github.com/raraz15/Discogs-VINet), Araz et al. *ISMIR 2024* | 512-D audio version embeddings | Check that proposed pair sounds like a version, and reject obvious metadata collisions. citeturn785950view0turn785950view1 |

A robust pairing pipeline is therefore:

**Reported-source stage:** select distinct Discogs-VI versions inside one clique, retain release identifiers, release dates/countries, artists and track metadata, then cross-reference to MusicBrainz/SHS where available. Discogs-VI's construction already used writer and title metadata to identify musical versions. citeturn153862search1turn785950view0

**Interpretation/filter stage:** reject candidate pairs when they appear to be merely another pressing, format, repress, reissue or remaster of the same recording. Discogs explicitly treats different formats/countries/pressings as separate Release records and groups related editions under a Master Release; the same recording can therefore yield multiple country/year records without being a new rendition. citeturn629722search3turn629722search11

For each candidate `(A,B)`, require at minimum: different performer or independently verified distinct recording identity; same Discogs-VI clique or same SHS/MB Work; dates separated by your desired interval; and audio-version similarity above a conservative threshold. Then annotate provenance as `composition_id`, `recording_A`, `recording_B`, `release_A`, `release_B`, `first_release_date_A/B`, `release_country_A/B`, and `evidence_sources`. This schema is my recommendation, not a published Discogs convention.

For reissues, **MusicBrainz offers the cleaner semantic test**: if two market records point to the same Recording MBID, treat them as the same recording unless contrary evidence exists. MusicBrainz defines recordings separately from releases and works precisely for this distinction. citeturn268313search0turn268313search6

For Discogs-only material, use format descriptions (`Reissue`, `Remastered`, `Repress`) as negative evidence, but do not assume those tags are complete. Discogs documents these edition descriptors, and its Master Release concept intentionally groups multiple releases without asserting that they are distinct performances. citeturn629722search0turn629722search3

**Recommended route for this team.** Build a high-precision gold set before scaling: require agreement from at least two of Discogs-VI, MusicBrainz Work links and SecondHandSongs for the evaluation pairs; use Discogs-VINet or another VI encoder only as a *filter*, not as the source of ground truth. This prevents your generation evaluation from inheriting the same model's mistakes.

**Open risks.** Audio similarity cannot alone distinguish a remaster from the original recording, because remasters can remain extremely close in melodic and structural content. Conversely, genuine covers may differ substantially in tempo, arrangement and key. The metadata graph must remain primary, with the embedding acting as supporting evidence. That conclusion follows from the entities defined by Discogs/MusicBrainz and the purpose of VI embeddings. citeturn629722search3turn268313search0turn785950view0

## Era and market prediction from audio

### Verified systems and results

| System / paper | Authors, venue/year | Task and reported result | Encoder/features | Relevance |
|---|---|---|---|---|
| **Music Era Recognition Using Supervised Contrastive Learning and Artist Information** | Qiqi He, Xuchen Song, Weituo Hao, Ju-Chiang Wang, Wei-Tsung Lu, Wei Li; arXiv 2024 — no peer-reviewed venue verified | Audio-only model reports **54% accuracy within ±3 years** on MSD; artist information adds 9 percentage points | Supervised-contrastive audio representation plus optional artist representation | Closest recent directly verified era-recognition paper. [DOI](https://doi.org/10.48550/arXiv.2407.05368) citeturn165899academia19 |
| **Production Effect: Audio Features for Recording Techniques Description and Decade Prediction** | Damien Tardieu, Emmanuel Deruty, Christophe Charbuillet, Geoffroy Peeters; DAFx 2011 | Decade classification of Pop/Rock using production-related cues | Dynamic range, inter-channel energy difference, phase spread | Direct evidence that mastering/production carries era signal. [paper page](https://www.dafx.de/paper-archive/details/Gk_oA1v_0dxmNhhBVBRISg) citeturn995251search2 |
| **YearPredictionMSD** | Thierry Bertin-Mahieux; MSD/UCI 2011 | 515,345 examples, 1922–2011, release-year regression benchmark | Echo Nest timbre features | Historical baseline corpus; not a modern waveform encoder. [DOI](https://doi.org/10.24432/C50K61) citeturn995251search0turn995251search8 |
| **Geographical Origin Prediction of Folk Music Recordings from the UK** | Vaiva Kedyte, Maria Panteli, Tillman Weyde, Simon Dixon; ISMIR 2017 | Geographic-coordinate prediction; average error reported below 120 km | Hand-engineered musical/audio feature vector + neural net | Demonstrates audio contains geographic signal, but on folk field recordings, not commercial market labels. [paper record](https://openaccess.city.ac.uk/id/eprint/19290/) citeturn995251search1 |
| **From Sound to Map: Predicting Geographic Origin of Musical Compositions** | Dominik Kostrzewa, Piotr Grabczyński; ICCS 2024 | **60.03% country accuracy on 44 country classes**; 72.42% continent, 60.26% subregion | Classical ML over extracted musical/audio descriptors | Strong verified country-origin result, but target is cultural origin, not release market. [DOI](https://doi.org/10.1007/978-3-031-63751-3_12) citeturn995251search9 |

There is **no verified standardized SOTA benchmark I found for “commercial release market from waveform”** analogous to ImageNet classification. The available geography studies typically predict cultural/geographical origin; this is a materially different target from “released in Spain versus released in the United States.” citeturn995251search1turn995251search9 Your own weak accompaniment-only market accuracy is therefore unsurprising: market may not be a stable acoustic property at all when the release is the same recording.

**Reported confounds.** Tardieu et al. explicitly use dynamic-range and stereo-production features for decade prediction, meaning an era classifier can learn mastering and recording technology instead of instrumentation or composition. citeturn995251search2 YearPredictionMSD is dominated by contemporary Western commercial music and has a heavily nonuniform year distribution, while its prescribed split tries to limit artist leakage. citeturn995251search0 Country-origin datasets are often traditional/world-music collections, so their signal can combine instrumentation, tuning, repertoire and recording provenance rather than a clean market variable. citeturn995251search5turn995251search9

**Interpretation for Time Traveler.** Your existing observation—that era is easier from accompaniment while market is more evident from vocals/language—is scientifically plausible, but the important methodological move is to define *market* explicitly. I recommend “release country of a distinct recording” as the label, and separately record `language`, `artist_area`, `genre/style`, and `label` as nuisance variables. A model should be tested both with and without vocals and with matched-language subsets.

For era, report both classification and regression: exact decade accuracy, ±1-decade accuracy, MAE in years and calibration. For market, use macro-F1 and balanced accuracy, not just ordinary accuracy, because country distributions will be unequal.

**Recommended route for this team.** Keep your codec-matched MERT/MuQ probes, but train a completely independent family of era/market evaluators—e.g. one SSL music encoder probe and one production-feature baseline—on disjoint artists and preferably disjoint compositions. Agreement across model families is more convincing than a single high probe score.

**Open risks.** A classifier may call something “1980s” because it is louder, narrower/wider in stereo, contains a gated snare, or has a particular codec/mastering signature. A country classifier may mainly be a language detector. These are not necessarily failures if the perceptual target is “sounds as if released there/then,” but they must be reported as possible causal pathways rather than assumed latent concepts. citeturn995251search2turn995251search9

## Controllable cover generation and style transfer

| System | Authors / venue | Conditioning mechanism | What it preserves / changes | Main limitation for Time Traveler |
|---|---|---|---|---|
| **MusicGen / MusicGen-Melody — “Simple and Controllable Music Generation”** | Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez; NeurIPS 2023 | Autoregressive EnCodec-token LM conditioned on text and optional melody representation | Melody condition gives structural control while text steers style/instrumentation | Not explicitly disentangled into era and market; vocals/lyrics are not reliably preserved. [paper](https://papers.neurips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html) citeturn362072search0turn296508search4 |
| **JASCO — “Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation”** | Or Tal, Alon Ziv, Itai Gat, Felix Kreuk, Yossi Adi; ISMIR 2024 | Flow matching with information-bottleneck conditioning; chords, melody, drums/full-mix plus global text | Fine local temporal controls plus global style text | Instrumental/general music generation rather than lyric-preserving cover synthesis. [DOI](https://doi.org/10.5281/zenodo.14877325) citeturn362072search9turn362072academia50 |
| **Combining Audio Control and Style Transfer Using Latent Diffusion** | Nils Demerlé, Philippe Esling, Guillaume Doras, David Genova; ISMIR 2024 | Disentangles local structure from global timbral/style information using diffusion-autoencoder representations | Demonstrates cover-like transfer of rhythmic/melodic content into another genre/style | Instrumental focus; not designed around lyrics or country. [DOI](https://doi.org/10.5281/zenodo.14877437) citeturn362072search1turn362072search5 |
| **Stable-Audio-Control — “Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer”** | Siyuan Hou et al.; arXiv 2024 | DiT + ControlNet; text plus top-k CQT melody condition | Style editing while explicitly retaining melody | Open-source instrumental training/evaluation; lyrics not central. [arXiv](https://arxiv.org/abs/2410.05151) citeturn296508academia10 |
| **Instruct-MusicGen** | Yixiao Zhang et al.; ISMIR 2025 | MusicGen plus LoRA text fusion and an audio-fusion adapter; instruction tuning | Adds/removes/separates stems while using original audio as condition | Published tasks are stem operations, not historical/country transfer. [paper](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2025/ZhangEtAl-ISMIR2025-InstructMusicGen.pdf) citeturn296508search9turn296508search5 |
| **Melodia — “Training-Free Music Editing Guided by Attention Probing in Diffusion Models”** | Y. Yang et al.; AAAI 2026 | Training-free reuse/manipulation of self-attention in AudioLDM2 | Targets genre/instrument/mood edits while preserving temporal structure | Source structure can be preserved better, but exact lyric identity is not guaranteed. [DOI](https://doi.org/10.1609/aaai.v40i3.37204) citeturn296508search8 |
| **MuLaCover — “Flexible Cover Song Generation via Symbolic Tonal Control”** | Lynn Yi, Zihan Xiong, Junjie Cao et al.; arXiv 2026 | Symbolic lead sheet (melody + chords) injected with gated adaptive cross-attention into a pretrained text-to-song model; lyrics/style supplied separately | Explicitly targets cover generation while retaining tonal identity and allowing tempo/style/structure changes | Very new; peer-reviewed venue **UNVERIFIED** as of the opened source. [arXiv](https://arxiv.org/abs/2610.26684) citeturn362072academia47turn362072search2 |
| **FreeSVC — zero-shot multilingual singing voice conversion** | ICASSP 2025 | SPIN content representation, language embeddings and ECAPA2 speaker encoder | Intended to preserve lyrics/melody while altering singer identity cross-lingually | Converts voice identity; not a market/era production model. citeturn296508search7 |

MuLaCover is especially relevant conceptually because it explicitly models a cover as preserving melody/harmony while changing rendition attributes. Its curriculum uses mismatched renditions so that symbolic tonal content can come from one version while style/audio comes from another. citeturn296508search2turn362072academia47

MusicGen-Melody remains a reasonable engineering baseline for your current accompaniment-only architecture because it already accepts melody and textual style conditions. The important limitation is that “1980s Spanish pop” is an entangled textual description rather than two independently controllable latent variables. citeturn362072search0

I did **not** find a verified peer-reviewed system whose task is exactly “take an existing commercial song and render it as if released in year Y and country C while preserving its melody and lyrics.” The closest verified lines are melody-conditioned music generation, cover generation, genre/timbre transfer, music editing, and multilingual singing-voice conversion. That exact formulation remains a credible research gap based on the sources above.

**Interpretation for architecture.** The evidence favors factoring your generator similarly to your probes: one condition/adaptor on accompaniment/production for era, and a separate vocal-side condition for market/language/accent. A single global “country+year” style vector would make leakage harder to diagnose.

**Recommended route for this team.** Preserve your working MusicGen-Melody accompaniment path as the era branch, and develop market control as a second stage rather than forcing accompaniment to encode market. That can initially be vocal timbre/language control without claiming full cultural-national style transfer.

**Open risks.** “Spain” is not a single acoustic style. Market can collapse into Spanish language, specific genres, or artist demographics. Similarly, an era prompt can collapse into a few cliché instruments. Human evaluation needs to ask separately about historical plausibility, geographic/market plausibility and stereotypicality.

## Evaluating a two-axis move

| Evaluation dimension | Recommended measure | Why |
|---|---|---|
| **Era movement** | Independent era probes; expected-year regression; decade classifier; human forced-choice between source and target era | Measures direction and magnitude rather than only target-class hit rate |
| **Market movement** | Independent market classifier with macro-F1/balanced accuracy; language-conditioned and instrumental-only versions; human country/region choice | Separates market signal from language |
| **Axis leakage** | Hold target market fixed while moving year, and vice versa; report Δ of the *non-target* probe | Direct test of disentanglement |
| **Melody retention** | Pitch-contour/chroma similarity plus version-identification retrieval | Protects same-song identity while tolerating key/tempo changes |
| **Lyrics retention** | ASR transcript similarity / word-error rate on separated vocals plus human lyric intelligibility | Critical if lyrics are claimed invariant |
| **Overall fidelity** | Human A/B ratings, audio-quality ratings and embedding-distribution metrics | Probe scores alone cannot guarantee musical quality |

**Reported foundation.** Discogs-VINet and related cover-identification systems are explicitly designed to recognize versions despite rendition changes, so VI retrieval is a natural invariant-content metric. Araz et al. evaluate with MAP and first-relevant rank. citeturn785950view0 MusicGen, JASCO, Mustango and related generation work all combine objective conditioning-adherence metrics with human evaluation rather than treating one classifier as sufficient. citeturn362072search0turn362072search9turn528705search7

A particularly useful design is a **2×2 counterfactual grid** per source:

`(original era, original market)`, `(target era, original market)`, `(original era, target market)`, `(target era, target market)`.

That design lets you estimate the main effect of each transformation and their interaction. It is my recommended experimental design, not a reported result.

For each generated clip, calculate:

\[
\Delta_E = s_E(y_{target}) - s_E(y_{source})
\]

for the era score and

\[
\Delta_M = s_M(m_{target}) - s_M(m_{source})
\]

for market, plus leakage terms such as the market-score change under an era-only edit. Report distributions and confidence intervals rather than only mean accuracy.

**Circular-evaluation risk.** If the adapter is trained to maximize a MERT-based era direction and the evaluation probe is another linear head on the same MERT representation trained from the same clips, success may mean only that the generator learned that representation's decision boundary. This is an interpretation of the experimental dependency, but it is analogous to the broader benchmark-overlap concern explicitly discussed by Araz et al., who avoided comparisons when training/test overlap could make results unreliable. citeturn785950view0

A stronger protocol is therefore: train control using encoder family A; evaluate era with B and C; evaluate song identity with an unrelated VI model; and run humans on a stratified sample. No artist, composition, or exact master/reissue family should cross the training/evaluation boundary.

**Human-test recommendation.** Do not ask “Does this sound Spanish?” alone. Ask independent questions: target-era plausibility; target-market plausibility; same-song identity; melody preservation; lyric preservation; naturalness; and confidence. Include source, real target-market/era references, and generated controls.

**Recommended route for this team.** Your most publication-ready next evaluation is the 2×2 intervention test with **two independent encoders plus humans**, because it directly demonstrates that year can move while market stays fixed and vice versa.

**Open risks.** The source and target labels themselves are noisy, probes can exploit mastering artifacts, and human raters may rely on stereotypes. These should be described as measurement limitations rather than hidden behind aggregate accuracy.

## Legal and ethical boundaries

| Material | Safest research release | Main source |
|---|---|---|
| Discogs-VI metadata | Redistribution subject to CC BY-NC-SA 4.0 and attribution/share-alike terms | MTG dataset page. citeturn393202search0 |
| MusicBrainz core metadata | Can be redistributed under CC0 | MusicBrainz license. citeturn268313search0turn268313search1 |
| YouTube IDs / URLs | IDs/metadata can be referenced subject to source terms; do not imply rights to audiovisual content | YouTube terms distinguish access to service/content; Discogs-VI distributes IDs rather than audio. citeturn589583search0turn153862search2 |
| Downloaded commercial audio | Do **not** redistribute absent permission or a jurisdiction-specific exception permitting it | YouTube prohibits unauthorized downloading/use; Discogs-VI does not redistribute audio. citeturn589583search0turn153862search2 |
| Extracted Discogs-VI CQT/chroma/HPCP | Use under MTG's research-only access conditions; redistribution rights should be confirmed with MTG | MTG says representations are available on request for non-commercial scientific research. citeturn393202search0turn785950view0 |
| Code and trained adapters | Normally safest artifact if it contains no redistributable copyrighted recordings; honor upstream code/model licenses | Discogs-VI code AGPLv3; Discogs-VINet repo MIT. citeturn393202search0turn785950view1 |
| Generated cover clips | Rights depend on whether protected musical-work or sound-recording expression remains recognizable and on jurisdiction; no blanket “AI-generated = free to publish” rule | U.S. Copyright Office says use of another musical work usually requires permission or an applicable exception; fair use is case-specific. citeturn805922search0turn805922search3 |

YouTube's public terms are the clearest practical obstacle to treating YouTube-linked datasets as ordinary downloadable research corpora. The Terms prohibit downloading content except when authorized by the service or permission, and API policies specifically prohibit storing audiovisual content without prior written approval. citeturn589583search0turn589583search1

EU Directive 2019/790 Article 3 creates a scientific-research text/data-mining exception for qualifying research organizations with lawful access, and Discogs-VI explicitly says UPF relied on that framework. citeturn589583search4turn153862search2 Whether that exception overrides or affects platform contractual restrictions is a legal question that depends on jurisdiction and institutional circumstances; the dataset documentation does not establish a universal answer.

Because your timezone/context is Japan, Japanese guidance is particularly relevant. Japan's Agency for Cultural Affairs states that Copyright Act Article 30-4 can permit reproduction for information analysis when the purpose is non-enjoyment, but it also warns that intentionally training so as to reproduce creative expression can fall outside that rule. The Agency explicitly says the guidance is interpretive, not a binding judgment for every AI system. citeturn805922search4turn805922search5turn805922search6

The **generation/publication stage is legally distinct from training**. Japanese Agency materials explicitly separate AI-development/training uses from generation/utilization, including public upload of outputs. citeturn805922search9 In the United States, the Copyright Office likewise emphasizes that using someone else's musical work or sound recording generally requires permission, a license, public-domain status or a valid exception; there is no fixed “safe number of seconds.” citeturn805922search3turn805922search0

**Interpretation for this project.** A generated clip intentionally preserving the source melody and lyrics is precisely the case where publication risk is higher than publishing an abstract embedding or evaluation score. The fact that the production/era has been transformed does not erase the underlying composition rights.

For a paper release, the conservative artifact set is therefore: code; model/adaptor weights if upstream licenses allow; train/evaluation IDs; metadata-derived pair lists where licenses allow; numerical features only where redistribution is explicitly permitted; probe predictions; and spectrogram/aggregate figures where appropriate. For commercial-source examples, host neither source audio nor generated full-song covers without institutional clearance. Short generated examples can still raise rights questions, so “30 seconds” should not be treated as a legal safe harbor. citeturn805922search0turn805922search3

**Ethical risk.** Country-conditioning can inadvertently encode language, ethnicity or cultural stereotypes. Report what the control actually changes—language, instrumentation, mix, vocal timbre, rhythm—rather than claiming that a model has learned a singular “Spanish sound.” That is an interpretive research-ethics recommendation.

**Recommended route for this team.** Before any public demo containing recognizable commercial compositions, obtain a short written determination from the university's copyright/legal office specifying what generated excerpts may be published. Until then, keep commercial-source generations inside the research environment and release IDs, code and quantitative outputs instead.

**Open risks.** Applicable rights differ across musical composition, sound recording, performance and possibly voice/personality rights; platform contract restrictions are separate from copyright exceptions; and AI-output law continues to evolve. The U.S. Copyright Office's 2025 AI report also confirms that copyrightability of generated output depends on sufficient human authorship rather than merely prompting a model. citeturn805922search1turn805922search2

## Ranked data routes for Time Traveler

| Rank | Route | Why it is strongest | First concrete step |
|---|---|---|---|
| **First — Discogs-VI metadata + MTG representations + MusicBrainz validation** | Discogs-VI already gives approximately 1.9M versions grouped into about 348k works and retains rich release metadata; MusicBrainz supplies an independent Work/Recording/Release model with explicit release country/date. citeturn785950view0turn268313search0 | Best combination of scale, same-composition grouping, year/country information and a principled reissue filter. Audio itself remains the access problem. | **Download `main.zip` from the official Discogs-VI Zenodo release and email MTG requesting the Discogs-VI-YT chroma/HPCP/CQT representations for non-commercial academic research.** citeturn153862search2turn393202search2 |
| **Second — MusicBrainz Work graph + legally available audio sources** | MusicBrainz core data is CC0 and explicitly separates Work, Recording, Release and Release Group, making it excellent for preventing “same recording, different reissue” false pairs. citeturn268313search0turn268313search4 | Metadata is legally easy to redistribute; audio coverage must be sourced separately, ideally from Creative-Commons catalogs or institutionally licensed holdings. | **Load the MusicBrainz core dump and query distinct Recording MBIDs linked to one Work where the recordings' earliest Releases differ in both year and country.** citeturn268313search2turn268313search6 |
| **Third — Da-TACOS / SecondHandSongs as a high-confidence cover graph, enriched with MusicBrainz** | Da-TACOS has explicit SHS Work IDs and release years, and 25k songs across benchmark/analysis subsets; the official SHS-100K 2025 set provides 100k performances grouped into 10k works with YouTube IDs. citeturn726336view2turn726336view1 | Very strong composition identity, but country is not in the released SHS-100K CSV and Da-TACOS does not include country, so market labeling requires enrichment. | **Take Da-TACOS's provided MusicBrainz candidates and resolve each performance to a distinct Recording MBID, then attach earliest Release date/country and discard same-recording reissues.** citeturn726336view2turn268313search7 |

For a three-person team with a GPU server and no budget, the **first route is substantially ahead of the alternatives**. It directly complements the course's existing Discogs-VI subset, minimizes a costly new matching effort, and lets the core scientific contribution shift from “building another cover dataset” to the harder and more defensible problem: disentangled temporal and market transformation with explicit invariance tests. The principal unresolved dependency is lawful audio access; the metadata, pair graph and feature-request path are already verified. citeturn153862search2turn785950view0