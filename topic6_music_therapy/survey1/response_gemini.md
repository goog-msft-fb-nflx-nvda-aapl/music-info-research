# **Transferring EEG-Based Emotion Modelling Across Datasets for Emotion-Conditioned Music Accompaniment**

## **Executive Summary**

The investigation into the proposed pipeline for developing an emotion-conditioned music accompaniment system reveals a landscape characterized by emerging algorithmic capabilities but constrained by severe methodological and data-centric bottlenecks. The analysis evaluates the feasibility of cross-dataset electroencephalography (EEG) transfer from the Database for Emotion Analysis using Physiological Signals (DEAP) to the Naturalistic Music EEG Dataset \- Tempo (NMED-T) to condition a latent diffusion audio generation model. The findings dictate critical architectural pivots to ensure empirical validity.  
The ten most important findings of this research are as follows:

> 1. **Systemic Evaluation Leakage in DEAP Benchmarks:** A vast majority of literature reporting classification accuracies exceeding 85% on the DEAP dataset relies on segment-level, trial-level, or mixed-subject cross-validation1. These protocols introduce severe data leakage because temporal baseline signatures of individual subjects are mixed across training and testing folds. Proper Leave-One-Subject-Out (LOSO) cross-validation yields significantly more grounded accuracies, typically around 53–65% for arousal and valence using modern architectures3.  
> 2. **Temporal Granularity Mismatch in Continuous Regression:** Regressing continuous valence-arousal (VA) trajectories on DEAP is inherently flawed from a ground-truth perspective because the dataset only provides a single discrete self-assessment manikin (SAM) rating (1–9) per 60-second trial6.  
> 3. **Superiority of the Concordance Correlation Coefficient:** When evaluating regression on emotion coordinates, the Root Mean Square Error (RMSE) metric heavily penalizes outliers and fails to capture trajectory covariance. The Concordance Correlation Coefficient (CCC) is mathematically superior for continuous emotion recognition as it measures both covariance and mean-shift alignment9.  
> 4. **Maturation of Open-Weight EEG Foundation Models:** Pretrained EEG models have reached viability for feature extraction. Models such as LaBraM (1.04B parameters, utilizing masked autoencoding)11 and CBraMod (\~4M parameters, utilizing criss-cross attention)13 offer highly robust starting points that outperform from-scratch convolutional neural networks.  
> 5. **The NMED-T Audio Deficit:** The NMED-T dataset provides high-quality 128-channel EEG but explicitly does not distribute the original stimulus audio, providing only metadata links15. This is a critical mechanical impediment for an audio-to-audio diffusion pipeline.  
> 6. **Catastrophic Overfitting Risk on Small Data:** NMED-T contains only 10 unique songs15. Fine-tuning a parameter-heavy diffusion model, such as a 1.2 billion parameter Diffusion Transformer (DiT), on 10 audio contexts guarantees catastrophic memorization, preventing the model from learning generalized emotional accompaniment.  
> 7. **Unverified Cross-Dataset Continuous Transfer:** Transferring an encoder trained on DEAP (which utilizes music-video stimuli and a 32-channel montage) to NMED-T (which utilizes purely auditory electronic dance music stimuli and a 128-channel montage) entails extreme modality, hardware, and task shifts. No prior work exists that explicitly validates zero-shot DEAP-to-NMED-T continuous emotion transfer.  
> 8. **Viability of Latent Diffusion Backbones:** stabilityai/stable-audio-open-1.0 (1.2B parameters) is an optimal open-weights candidate. It supports 44.1kHz stereo generation up to 47 seconds, utilizing an autoencoder and a DiT architecture with cross-attention mechanisms that can be adapted for EEG/VA conditioning tokens18.  
> 9. **Circularity in Generative Evaluation:** Evaluating generation alignment using an audio-to-emotion predictor that was trained on the same pseudo-labels used to condition the generative model creates an invalid circular evaluation loop. Independent models, such as MERT-v1-95M combined with Frechet Audio Distance (FAD), must be utilized to maintain objective rigor20.  
> 10. **Legal Protections for Text and Data Mining:** Japan's Copyright Act Article 30-4 provides a highly robust exception for text and data mining (TDM). It explicitly allows the reproduction and exploitation of copyrighted works for machine learning model training without rightsholder permission, provided the usage is for "non-enjoyment" purposes22. This legal framework allows researchers to independently scrape the copyrighted NMED-T audio stimuli.

The assessment of the proposed pipeline phases is as follows. Phase 1 (DEAP Encoder Training) is a **Go**. Training an EEG encoder on DEAP using LOSO cross-validation is highly feasible. Establishing a rigorous baseline using CCC on foundation models (e.g., CBraMod or LaBraM) will yield a strong standalone contribution to the literature. Phase 2 (Channel Harmonization and Pseudo-Labeling) is a **No-Go / High Risk**. Interpolating NMED-T 128-channel data to DEAP 32-channel data and extracting pseudo-labels will function mechanically, but the validity of these labels will be entirely unverified due to the massive distribution shift and the lack of ground-truth behavioral emotional data in NMED-T. Phase 3 (Emotion-Conditioned Accompaniment Generation) is a **No-Go as proposed**. Conditioning a diffusion model on NMED-T is fundamentally broken because 10 songs provide insufficient acoustic diversity. The generative model will overfit to the stimulus identity rather than learning generalized emotional accompaniment.  
The five most dangerous risks to the project are: (1) Overfitting a latent diffusion audio backbone to the limited 10 stimuli in NMED-T; (2) The inability to practically source the exact uncompressed audio for NMED-T to match the EEG temporal alignment; (3) Reporting inflated Phase 1 results by accidentally leaking subject data in temporal splits; (4) The likelihood that pseudo-labels will reflect the DEAP label distribution rather than the actual emotional state of NMED-T subjects; and (5) Evaluating audio generation quality using metrics (like FAD with VGGish) that fail to capture nuanced musical and emotional coherence.

## **1\. EEG Emotion Recognition on DEAP**

The Database for Emotion Analysis using Physiological Signals (DEAP) serves as the primary benchmark for EEG-based emotion recognition. The dataset features 32 participants who each watched 40 one-minute music videos. Physiological signals were recorded using a 32-channel EEG montage, and participants provided per-trial self-assessment manikin (SAM) ratings on a discrete 1 to 9 scale for valence, arousal, dominance, and liking6.

### **Dominant Architectures and the Evaluation Protocols**

Since 2018, deep learning architectures have superseded traditional machine learning approaches for DEAP evaluation. Early approaches relied on Support Vector Machines (SVM) or Random Forests applied to extracted features such as differential entropy, band power across discrete frequency bands (theta, alpha, beta, gamma), and Hjorth parameters24. Contemporary approaches leverage convolutional neural networks (CNNs), graph neural networks (GNNs), and transformer-based foundation models to learn spatial-temporal representations directly from raw or minimally processed signals.

| Title | Authors | Year | Venue | Task / Data | Key Numbers (with protocol) | Code Link | Verified |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition | Ding et al. | 2022 | IEEE TAC | DEAP / Val & Aro Class. | LOSO: Val 0.54±0.09 F1, Aro 0.56±0.20 F1 | github.com/EmotionLab/EEGain | Yes |
| LGGNet: Learning from Local-Global-Graph Representations for Brain-Computer Interface | Ding et al. | 2023 | IEEE TNNLS | DEAP / Emotion Class. | Segment-level CV: Val/Aro Accuracy \> 80% | github.com/yi-ding-cs/LGG | Yes |
| EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization | Song et al. | 2023 | IEEE TNSRE | DEAP / Emotion Class. | Subject-Dependent CV: Accuracy \~90% | UNVERIFIED | No |
| LaBraM: Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI | Jiang et al. | 2024 | ICLR | DEAP / Emotion Class. & Reg. | Multi-Subject CV: Accuracy 96-97% | github.com/935963004/LaBraM | Yes |
| CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding | Wang et al. | 2025 | ICLR | DEAP / Downstream BCI | Multi-Subject CV: Competitive SOTA | github.com/wjq-learning/CBraMod | Yes |
| BIOT: Biosignal Transformer for Cross-data Learning | Yang et al. | 2023 | NeurIPS | Cross-dataset tokenization | Outperforms baseline CNNs | github.com/ycq091044/BIOT | Yes |

A critical analytical distinction must be made regarding the evaluation protocols reported in the literature. Papers reporting classification accuracies exceeding 85% on the DEAP dataset almost universally employ segment-level, trial-level, or mixed-subject cross-validation protocols1. EEG signals possess a highly subject-specific baseline, often referred to as an individual neural fingerprint. If short temporal windows from the same subject appear in both the training and testing folds, the model inadvertently learns to identify the subject's baseline state rather than the generalized neural correlates of emotion. This constitutes severe data leakage. Proper Leave-One-Subject-Out (LOSO) protocols, which test the model on a completely unseen subject, yield significantly lower but realistic accuracies. Under LOSO, binary classification accuracies generally regress toward 50–65%, with F1 scores in the low 50s for models like TSception and EEGNet3.  
Among the strongest architectures, TSception utilizes multi-scale 1D convolutional kernels in a dynamic temporal layer to extract time-frequency representations, paired with an asymmetric spatial layer that captures inter-hemispheric discrepancies related to emotion processing27. LGGNet models the complex relations within and among functional brain areas using a neurologically inspired local-global graph neural network, though its reported high accuracies heavily rely on segment-level cross-validation rather than rigorous LOSO28. Foundation models present the most promising path forward. LaBraM (1.04B parameters) utilizes vector-quantized neural spectrum prediction and masked patch reconstruction on massive clinical datasets to learn generic EEG representations11. CBraMod (\~4M parameters) offers a highly efficient alternative, utilizing criss-cross attention to separately model spatial (channel) and temporal (patch) dependencies, proving highly adaptable for fine-tuning on downstream tasks like DEAP13.

### **Continuous Regression and Label Granularity**

Regressing continuous VA trajectories on DEAP presents a fundamental temporal mismatch. DEAP annotations are per-trial, meaning a single integer rating is applied to a full 60-second exposure6. Applying continuous regression objectives to short EEG windows (e.g., 2-second overlapping segments) implicitly assumes that the subject's emotional state was static across the entire minute. This assumption violates the dynamic nature of music-induced emotion.  
When evaluating regression, standard metrics include the Root Mean Square Error (RMSE), the Pearson Correlation Coefficient (PCC), and the Concordance Correlation Coefficient (CCC)9. The RMSE is highly sensitive to outliers because errors are squared before averaging8. On the 1-9 DEAP scale, standard LOSO regression models report RMSE values typically between 1.90 and 2.11 for valence and arousal7. Claims of RMSE below 0.16 in some literature are anomalous and usually result from min-max normalization mapping the 1-9 scale to a 0-1 range prior to error calculation, rendering direct numerical comparisons invalid25.  
The Concordance Correlation Coefficient (CCC) is mathematically superior to both RMSE and PCC for continuous emotion recognition. While PCC only measures linear correlation, failing to penalize predictions that are perfectly correlated but shifted in absolute value, CCC evaluates both covariance and mean-shift alignment. It measures whether the predictions follow the same trajectory and inhabit the same absolute numerical range as the ground truth9. The equation for CCC is ![][image1], where ![][image2] is the Pearson correlation, and ![][image3] and ![][image4] represent the means and variances of the predicted and ground-truth series.

### **Preprocessing Standards**

Standard preprocessing for DEAP relies on the publicly available .dat and .mat files provided by the dataset authors. These files are downsampled to 128 Hz, bandpass filtered (typically 4-45 Hz to isolate Theta, Alpha, Beta, and Gamma bands while excluding slow drifts and line noise), and referenced to a common average32. A common pitfall in DEAP preprocessing is failing to perform baseline removal. DEAP provides a 3-second pre-stimulus baseline recording for each trial. Subtracting baseline features from the trial features is strictly necessary to mitigate inter-subject baseline variance and isolate the stimulus-evoked emotional response34.

## **2\. Cross-Dataset / Cross-Montage EEG Transfer**

### **Channel Harmonization**

To utilize a DEAP-trained encoder (which expects 32 channels) to extract pseudo-labels from the NMED-T dataset (which utilizes a dense 128-channel array), the spatial dimensions must be harmonized. Spatial interpolation, utilizing tools such as MNE-Python's spherical spline interpolation, maps the dense 128-electrode array onto the sparse 10-20 standard montage used by DEAP. However, interpolation introduces spectral smoothing, reducing high-frequency spatial independence and potentially distorting the original signal geometry.  
Alternatively, channel-agnostic encoders natively handle variable topologies. The BIOT architecture tokenizes diverse biosignals, enabling cross-data learning with mismatched channels and variable lengths35. CBraMod also provides flexibility through its criss-cross attention mechanism, which can be configured to process different channel counts without requiring rigid spatial interpolation13. A third approach involves strict subset selection, directly discarding 96 channels to match the nearest 10-20 equivalents. While this preserves the raw spectral properties of the remaining channels, it wastes the high spatial resolution inherent to NMED-T.

### **Cross-Dataset Emotion Recognition**

Transferring learned emotional representations across datasets introduces massive domain shifts. The literature reveals that direct zero-shot transfer (e.g., training on DEAP and testing on SEED, DREAMER, or AMIGOS) yields near-random accuracy due to differing elicitation materials, different reference electrodes, varying sampling rates, and label distribution shifts34. Domain adaptation methods are heavily utilized to align feature distributions across datasets. Techniques such as Maximum Mean Discrepancy (MMD), Correlation Alignment (CORAL), and Domain-Adversarial Neural Networks (DANN) attempt to minimize the distance between the source and target domain feature spaces. For instance, Temporal-Spatial Convolutional Learning with Learning without Forgetting (TSCL-LwF) has been applied to cross-subject domain adaptation, achieving improved robustness34.  
However, applying these techniques to zero-shot cross-dataset continuous emotion regression remains largely unsolved. There is explicitly no prior work that successfully trains an emotion regressor on DEAP and applies it to predict valid, continuous emotional states on the NMED-T dataset. The divergence in stimuli is profound. DEAP utilizes multimodal music videos, which induce emotion heavily via visual cortex activation alongside auditory processing. NMED-T utilizes purely auditory electronic dance music. A DEAP-trained encoder will likely fail to capture purely auditory emotional correlations when deployed on NMED-T, as it will expect visual-evoked cortical dynamics that are absent in the target domain.

## **3\. NMED-T Specifics**

The Naturalistic Music EEG Dataset \- Tempo (NMED-T) from Stanford University focuses on cortical and behavioral responses to continuous music listening15.  
The dataset contains recordings from 20 adult participants who each listened to 10 full-length commercial electronic dance music (EDM) songs. These songs span various sub-genres and tempos, but all contain electronically produced beats in a duple meter15. Neural activity was recorded using a 128-channel dense-array EEG system. While the original sampling rate was 1000 Hz, the preprocessed and aggregated data provided by the authors is downsampled to 125 Hz. The raw recordings comprise approximately 39GB of data, while the cleaned, song-aggregated files total approximately 700MB15. Behavioral data in NMED-T is limited to sensorimotor synchronization (tapping responses) and basic song ratings17. Crucially, it does not contain the rich, continuous valence-arousal tracking comparable to DEAP or other emotion-specific datasets.  
The most critical limitation of NMED-T for generative modeling pipelines is that the stimulus audio is strictly not published15. The authors provide metadata links, but researchers must independently source the exact audio files and manually align them to the EEG recordings. The dataset is distributed under a Creative Commons CC-BY 3.0 or 4.0 license15.  
Prior work utilizing NMED-T has primarily focused on decoding music information directly from the brain. Studies have successfully formulated song identification as a 10-way classification problem, predicting the specific song ID from EEG segments38. Advanced approaches have extracted acoustic features using models like MuQ and predictive expectation features (surprisal and entropy) using MusicGen, demonstrating that EEG representations heavily encode structural and acoustic properties of the music when the context window exceeds approximately two seconds38.  
The risk of small-data generalization failure is acute. Ten songs paired with 20 participants is catastrophically small for conditioning a generative diffusion model. A 1.2 billion parameter diffusion transformer will rapidly learn the exact waveforms of the 10 target tracks from the latent embeddings. The risk of rote memorization, or shortcut learning, is extremely high. The model will bypass the intended emotion conditioning entirely, instead learning to map the subject's specific sensory-evoked potentials directly to the identity of the song38.

## **4\. Alternative and Complementary EEG \+ Music Datasets**

To mitigate the profound limitations of NMED-T, alternative datasets must be evaluated to identify viable sources of paired physiological and emotional data.

| Name | Participants | Stimuli (Music) | EEG Channels / Hz | Emotion Labels | Audio Available? | License / Access | Link | Verified |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **DEAP** | 32 | 40 Music Videos (1 min) | 32 / 512 Hz | VA, Liking (1-9) per trial | Meta only | EULA | eecs.qmul.ac.uk/mmv/datasets/deap/ | Yes |
| **NMED-T** | 20 | 10 EDM Songs (full-length) | 128 / 125 Hz | Tapping, basic ratings | No | CC-BY | purl.stanford.edu/jn859kj8079 | Yes |
| **NMED-H** | 12 | 4 Bollywood Songs | 128 / 125 Hz | Basic ratings | No | CC-BY | purl.stanford.edu/... | Yes |
| **MAHNOB-HCI** | 27 | 30 Emotion sequences | 32 / 256 Hz | VA, Dominance (1-9) | Meta only | EULA | mahnob-db.eu/hci-tagging/ | Yes |
| **DREAMER** | 23 | 18 Film clips | 14 / 128 Hz | VA, Dominance | Meta only | EULA | zenodo.org/... | Yes |
| **AMIGOS** | 40 | Film clips (short/long) | 14 / 128 Hz | VA, basic emotions | Meta only | EULA | eecs.qmul.ac.uk/mmv/datasets/amigos/ | Yes |
| **SEED** | 15 | 15 Film clips | 62 / 1000 Hz | Categorical (pos/neg/neu) | Meta only | EULA | bcmi.sjtu.edu.cn/\~seed/ | Yes |
| **DEAM** | N/A | 1802 Audio tracks | None (Audio only) | Continuous VA (2Hz) | Yes | Open | freemusicarchive.org/... | Yes |
| **PMEmo** | N/A | 794 Audio tracks | None (Audio only) | Continuous VA (2Hz) | Yes | Open | github.com/HuiZhangDB/PMEmo | Yes |

While datasets such as SEED, DREAMER, and MAHNOB-HCI provide robust physiological data, they rely primarily on film clips or emotional sequences rather than pure music. Datasets belonging to the NMED family (NMED-T, NMED-H, NMED-E) utilize complete musical works but lack robust emotional annotations40.  
The DEAM and PMEmo datasets lack physiological data entirely, but their time-resolved, continuous valence-arousal annotations, captured at a 2 Hz resolution, make them the paramount choice for training an audio-side emotion-conditioning regressor or evaluating the emotional alignment of generated audio41. The availability of high-quality, continuous ground-truth labels directly paired with redistributable audio makes them indispensable for the generative phase of the pipeline.

## **5\. Emotion-Conditioned and EEG-Conditioned Music Generation**

### **Brain-to-Music Prior Work**

Prior research in brain-to-music generation has relied heavily on functional Magnetic Resonance Imaging (fMRI) due to its high spatial resolution, which allows for the decoding of complex semantic and structural stimulus representations. Systems like Brain2Music reconstruct auditory stimuli from fMRI blood-oxygen-level-dependent (BOLD) signals. In contrast, EEG-conditioned generation is highly experimental. Recent works mapping EEG to discrete music tokens using models like MusicLM or MusicGen, or utilizing contrastive alignment (CLAP-style architectures), generally struggle with the low signal-to-noise ratio of EEG. These systems frequently end up reconstructing the low-level acoustic envelope or tempo rather than high-level semantic emotion.

### **Valence-Arousal Conditioned Generation**

Significant progress exists in emotion-conditioned audio generation without physiological signals. Frameworks such as LARA-Gen enable continuous emotion control for music generation models via latent affective representation alignment. This system utilizes a unified emotion predictor to facilitate objective evaluation of emotional controllability42. Similarly, the MusicEmotionModeling framework proposed by Dai et al. implements valence and arousal control through in-context learning frameworks43.

### **Latent Diffusion Audio Backbones**

To implement emotion-conditioned accompaniment transformation—conceptually similar to audio inpainting or Stochastic Differential Equation editing (SDEdit)—a robust latent diffusion model is required.

| Backbone | Parameters | Sample Rate | Context/Tokens | License | HF Link / Checkpoint |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Stable Audio Open 1.0** | 1.2B (DiT \+ T5) | 44.1 kHz | Up to 47 seconds | Stability Community | stabilityai/stable-audio-open-1.0 |
| **AudioLDM2-Music** | 300M | 16 kHz | 10 seconds | CC BY 4.0 | cvssp/audioldm2-music |

stable-audio-open-1.0 is uniquely suited for high-fidelity accompaniment generation. It comprises three primary components: a 156M parameter autoencoder that compresses raw waveforms into a manageable latent sequence, a 109M parameter T5 text embedding model for conditioning, and a 1.05B parameter transformer-based diffusion model (DiT) that operates in the latent space18. The DiT architecture accepts cross-attention conditioning tokens, making it mechanically straightforward to inject EEG embeddings or VA coordinates as additional context. Fine-tuning the DiT layers via Low-Rank Adaptation (LoRA) can be achieved efficiently on a shared 4x H200 server architecture44.  
Modality dropout is a critical technique for this pipeline. By replacing the VA or EEG embeddings with null tokens during training at a fixed probability rate (e.g., 10%), the model enables classifier-free guidance at inference. This allows the system to flexibly rely on joint conditioning (audio context \+ VA \+ EEG), unimodal conditioning, or purely unconditional generation19.  
Emotion-aware generation objectives extend beyond the standard Mean Squared Error (MSE) loss applied to noise prediction, as standard MSE does not guarantee semantic emotional alignment. Advanced objectives include auxiliary regression loss, where the generated (and differentiable) latent representations are passed through a frozen emotion regressor, applying MSE toward the target VA coordinates. Alternatively, InfoNCE loss can be adapted for contrastive alignment, pushing the generated audio embeddings closer to the target emotion or EEG embeddings in a shared latent space43.

## **6\. Music Emotion Recognition (Audio Side) for Evaluation**

To quantitatively evaluate the emotional alignment of the generative output without relying on subjective human listening tests, an automated audio-side Music Emotion Recognition (MER) model is required.

### **Pretrained Audio Models**

The MERT (Music undERstanding model with large-scale self-supervised Training) architecture, specifically the m-a-p/MERT-v1-95M checkpoint, is highly effective for this task. MERT-v1-95M is a HuBERT-based self-supervised music encoder containing 95 million parameters, distributed under an Apache 2.0 license20. It was trained on massive datasets using a masked language modeling (MLM) paradigm with in-batch noise mixture, processing 24 kHz audio and outputting 768-dimensional features at a rate of 50 Hz (20 ms per frame)20. When MERT representations are utilized to train downstream regression heads on the DEAM and PMEmo datasets, the models yield high R2 and CCC scores for continuous valence-arousal prediction41. Contrastive Language-Audio Pretraining (CLAP) models, while robust for text-audio alignment, are generally less granular for continuous VA coordinate prediction unless explicitly fine-tuned on dense time-resolved data.

### **Measuring Emotion Alignment**

The proposed protocol evaluates the CCC (or MSE) between the target VA coordinates fed into the diffusion model and the VA coordinates predicted from the generated audio by the independent MER model. However, cross-dataset reliability is notoriously poor in audio MER. An MER model trained exclusively on the acoustic properties of the DEAM dataset may fail to generalize to the specific synthesized timbres produced by the Stable Audio DiT model, yielding artificially poor alignment scores that reflect acoustic domain shift rather than true emotional misalignment.

### **Standard Generation Metrics**

Beyond emotion, the core musical quality and distribution fidelity must be evaluated using the Frechet Audio Distance (FAD). FAD relies on toolkits such as fadtk45. While VGGish embeddings are the historical standard for FAD computation, computing FAD over MERT-v1-95M or CLAP-music embeddings provides a significantly stronger assessment of musical coherence and stylistic fidelity20. A critical caveat is that FAD is highly unstable on small test sets. Evaluating generative fidelity on the 10-song NMED-T dataset will yield statistically insignificant FAD scores due to the minimal distribution coverage.

## **7\. EEG-Audio Alignment and EEG Encoders Usable as Conditioners**

If raw EEG embeddings are to be used alongside VA coordinates for cross-attention conditioning in the diffusion model, an encoder that outputs rich, temporally stable tokens is necessary.  
Foundation models such as CBraMod (\~4M parameters) and LaBraM (1.04B parameters) provide optimal tokenization mechanisms12. CBraMod is highly efficient and available under the Apache 2.0 license, with inference implementations available in both Rust/Burn and PyTorch13. Its criss-cross transformer natively outputs 200-dimensional embeddings that capture distinct spatial and temporal hierarchies. These embeddings can be projected via a simple linear layer to match the DiT's required cross-attention dimension (e.g., 1024 for Stable Audio Open).  
A foundational research question remains: What precise information does EEG carry beyond explicit valence-arousal states? Research on the NMED-T dataset demonstrates that song identity—specifically acoustic features, temporal surprisal, and melodic entropy—is highly decodable from EEG signals38. The profound risk in utilizing EEG embeddings as conditioning tokens is that the token acts as a high-bandwidth "shortcut" for stimulus identity rather than providing abstract emotional priors. The diffusion model may simply learn to reconstruct the specific acoustic envelope of the song associated with that EEG signature. Strict ablations, specifically a shuffled-EEG control where EEG tokens are randomized relative to the audio context, are necessary to prove that the EEG conditioning supplies generalized emotional information rather than simply leaking the target song's identity.

## **8\. Music Therapy Framing and Clinical/Ethical Context**

### **Clinical Framing**

The motivation for adaptive music biofeedback relies heavily on the clinical "iso-principle." This principle dictates that a therapeutic intervention should initially match the user's current emotional state (entrainment) and gradually modulate the music to shift the user to a target state, such as lower physiological arousal or positive valence. Valence and arousal are clinically sensible continuous variables for the regulation of stress, depression, and anxiety. However, because this research relies entirely on non-clinical, offline datasets containing healthy participants, claims of therapeutic efficacy must be strictly curtailed. The proposed system serves as a proof-of-concept for the technical computational mechanism of adaptive accompaniment generation, not as a validated clinical intervention.

### **Ethics and Legal Context**

Data protection protocols dictate that public datasets like DEAP and NMED-T are heavily anonymized, mitigating direct privacy risks. A more complex issue arises regarding copyright law for machine learning, as NMED-T relies on copyrighted commercial electronic dance music.  
A critical legal advantage is provided by Japan's Copyright Act. Following a 2018 amendment, Article 30-4 establishes a highly flexible exception allowing the reproduction and exploitation of copyrighted works for Text and Data Mining (TDM) and machine learning model training22. This exception applies regardless of the rights holder's permission and covers both commercial and non-commercial uses, provided the purpose is "non-enjoyment"—meaning the works are not being consumed for their intended aesthetic or informational value by a human, but rather processed by a machine46. This legal framework explicitly protects the research team's ability to scrape, analyze, and train models on the copyrighted NMED-T audio stimuli, assuming the audio can be independently sourced to match the metadata.

## **9\. Methodological Risks and Mitigations**

The proposed research pipeline exhibits several structural vulnerabilities that require rigorous mitigation strategies or outright reformulations.

> 1. **Risk: Temporal Mismatch in Label Granularity.** Assigning a single 60-second SAM rating from DEAP to every overlapping 2-second EEG window assumes emotional stasis.*Mitigation:* Utilize sequence-to-sequence modeling, leveraging temporal Transformers or LSTMs to predict a distribution of emotions over the course of the trial, anchoring the final aggregated mean to the 1-9 rating. Evaluate performance strictly using CCC rather than RMSE to account for trajectory covariance.  
> 2. **Risk: Label Shift and Scale Mismatch.** DEAP stimuli rely on multimodal music videos, while NMED-T stimuli are purely auditory. Pseudo-labels generated for NMED-T will reflect DEAP's narrow label distribution, potentially ignoring auditory-specific emotional nuances.*Mitigation:* Perform unsupervised domain adaptation, such as CORAL, aligning the latent covariance matrices of the DEAP EEG and NMED-T EEG feature spaces prior to the regression head to ensure distributional alignment.  
> 3. **Risk: Channel Interpolation Smoothing.** Interpolating 128 channels down to 32 destroys high-frequency localized cortical dynamics. *Mitigation:* Abandon spatial interpolation. Utilize a channel-agnostic foundation model like CBraMod or BIOT that tokenizes individual channels independently and can natively process variable input topologies13.  
> 4. **Risk: Shortcut Learning and 10-Song Overfitting (CRITICAL).** Ten songs cannot train a 1.2B parameter DiT to generalize. The model will learn to reproduce the 10 songs identically based on the latent representations.*Mitigation:* Abandon NMED-T for the generative modeling phase. The dataset lacks the requisite acoustic diversity to train or fine-tune a latent diffusion backbone without catastrophic memorization.  
> 5. **Risk: Circular Evaluation.** Evaluating the emotional alignment of the generated audio using a predictor trained on the same data used to extract the conditioning pseudo-labels introduces a circular logic loop.*Mitigation:* Train the audio MER evaluation model strictly on independent datasets, namely DEAM or PMEmo. Never expose the evaluation model to DEAP or NMED-T distributions.

### **Fallback Designs**

Given the fatal bottlenecks identified in Phase 2 and Phase 3, the following fallback designs are ranked by feasibility and empirical rigor:

* **Fallback 1 (Highest Feasibility & Novelty):** Abandon NMED-T entirely. Train the generative diffusion model strictly on DEAM and PMEmo audio, conditioned on their high-quality, continuous VA labels. In parallel, train an EEG-to-VA model on DEAP. At inference time, feed unseen DEAP EEG into the EEG-to-VA model to generate a predicted VA trajectory. Feed this predicted VA trajectory into the DEAM-trained audio generator. This bypasses the need for explicitly paired EEG-and-Audio data during training, relying instead on VA coordinates as the universal bridging modality.  
* **Fallback 2 (Moderate Feasibility):** Utilize DEAP's own music-video soundtrack audio. Extract the audio from the 40 DEAP videos. Train the DiT on these 40 tracks conditioned on the paired DEAP EEG. While this still suffers from severe small-data overfitting risks, it guarantees exact temporal label pairing and modality consistency.  
* **Fallback 3 (Lowest Feasibility):** Rely on Japanese copyright exemptions to scrape the NMED-T audio, and heavily augment the 10 songs (via pitch shifting, time stretching, and convolution reverb) to create 10,000 synthetic tracks for DiT fine-tuning. This artificially expands the dataset to prevent direct memorization but risks destroying the delicate temporal alignment between the original audio and the recorded EEG.

## **10\. Novelty Check and Positioning**

A thorough review of the literature from 2022 to 2026 reveals no existing papers that map continuous, transferred EEG pseudo-labels to a latent diffusion DiT for the purpose of music accompaniment generation.  
The closest existing works focus on different modalities. LARA-Gen (2025) provides continuous emotion control for music generation via latent affective alignment, but focuses exclusively on text and audio, lacking any physiological input42. The MusicEmotionModeling framework by Dai et al. (2026) implements VA-controlled in-context learning, but similarly ignores EEG43. Systems like Brain2Music focus on stimulus reconstruction from fMRI, not emotion-conditioned accompaniment. Therefore, the pipeline is highly novel, though methodologically fragile if executed as initially proposed.  
To maximize impact, the research should focus on the following core Research Questions (RQs):

* **RQ1:** Does continuous Valence-Arousal conditioning derived from a cross-dataset EEG encoder yield higher musical emotion alignment (measured by independent MERT-CCC) compared to random VA trajectories?  
* **RQ2:** In multimodal generative guidance, does joint conditioning (EEG embeddings \+ VA coordinates) outperform VA-only conditioning in preserving temporal musical coherence (measured by FAD)?  
* **RQ3:** What is the performance degradation (LOSO transfer gap) when evaluating EEG foundation models under strict zero-shot cross-dataset settings versus intra-dataset settings?

Appropriate venues for publication include the International Society for Music Information Retrieval (ISMIR), which expects 8-page submissions focused on music generation and audio evaluation protocols. Alternatively, the IEEE Transactions on Affective Computing (TAC) is suitable if the research focus pivots heavily toward the mathematics of DEAP-to-NMED-T domain adaptation.

## **11\. Resources to Locate**

The following resources, models, and repositories are required for the execution of the pipeline:

* **Hugging Face Models:**  
  * *Stable Audio Open 1.0:* https\://huggingface.co/stabilityai/stable-audio-open-1.0 (1.2B parameters, Stability Community License)18.  
  * *AudioLDM2-Music:* https\://huggingface.co/cvssp/audioldm2-music (300M parameters, CC BY 4.0)49.  
  * *MERT-v1-95M:* https\://huggingface.co/m-a-p/MERT-v1-95M (95M parameters, Apache 2.0)20.  
  * *CBraMod:* https\://huggingface.co/braindecode/CBraMod50.  
* **GitHub Repositories:**  
  * *CBraMod (Foundation Model Implementation):* https\://github.com/wjq-learning/CBraMod14.  
  * *LaBraM (Foundation Model Implementation):* https\://github.com/935963004/LaBraM12.  
  * *Frechet Audio Distance Toolkit:* https\://github.com/microsoft/fadtk45.  
  * *TSception Baseline Code:* https\://github.com/EmotionLab/EEGain4.  
* **Toolkits and Benchmarks:**  
  * *Braindecode / TorchEEG:* Essential for standardized dataset loading and foundation model inference52.

## **Recommended Plan**

Given the severe risks surrounding NMED-T's lack of audio and its restrictive size, the following optimized execution plan is recommended for the shared 4x H200 server infrastructure.  
The primary dataset for EEG-to-Emotion regression will be DEAP. The primary dataset for audio generation and MER evaluation will be DEAM and PMEmo. NMED-T will be discarded for the generative phase and retained solely for an analysis of zero-shot domain transfer. The core models will utilize CBraMod for EEG feature extraction, Stable Audio Open 1.0 for latent diffusion generation, and MERT-v1-95M for independent audio evaluation.  
The minimum experiment list is as follows:

> 1. **Phase 1 (Smoke Test):** Fine-tune CBraMod on DEAP to regress VA utilizing a CCC loss objective. Strict enforcement of LOSO cross-validation is mandatory to prevent segment-level leakage. Report the final CCC metrics.  
> 2. **Phase 3 Modification (Audio-Side Supervision):** Fine-tune Stable Audio Open 1.0 on the DEAM and PMEmo audio datasets, conditioned on their ground-truth continuous VA labels. Utilize cross-attention LoRA adapters. The expected failure mode is Out-Of-Memory (OOM) errors on the 4x H200s, which must be mitigated by enabling gradient checkpointing and deepspeed stage 2 optimization.  
> 3. **Inference Integration:** Feed test-subject EEG from DEAP into the trained CBraMod to generate a continuous VA trajectory. Feed this predicted VA trajectory (utilizing modality dropout on explicit EEG tokens) into the trained Stable Audio model to generate the accompaniment track.  
> 4. **Quantitative Evaluation:** Execute fadtk using MERT embeddings to ensure generated audio quality matches the target distributions. Run the MERT-MER regressor on the generated audio to predict VA, and calculate the CCC against the original CBraMod-predicted VA to quantify final alignment.

The initial download requirements for the shared server storage will include the Stable Audio Open weights (\~5GB), MERT-v1-95M weights (\~1.7GB), DEAP preprocessed .mat files (\~3GB), and the PMEmo dataset audio and labels (\~15GB). This totals approximately 25GB, which is highly feasible for a restricted shared storage environment.

#### **Works cited**

> 1. Multi-Scale Dynamic Temporal Network With Graph Matching, [https\://www\.researchgate.net/publication/401569408\_Multi-scale\_Dynamic\_Temporal\_Network\_with\_Graph\_Matching\_Domain\_Adaptation\_for\_Cross-Subject\_EEG\_Emotion\_Recognition](https://www.researchgate.net/publication/401569408_Multi-scale_Dynamic_Temporal_Network_with_Graph_Matching_Domain_Adaptation_for_Cross-Subject_EEG_Emotion_Recognition)  
> 2. MEEG and AT-DGNN: Improving EEG Emotion Recognition with, [https\://www\.computer.org/csdl/proceedings-article/bibm/2024/10821806/23oohU4pyMw](https://www.computer.org/csdl/proceedings-article/bibm/2024/10821806/23oohU4pyMw)  
> 3. Cross-Dataset Emotion Valence Prediction Approach from 4 ... \- MDPI, [https\://www\.mdpi.com/2504-2289/9/11/280](https://www.mdpi.com/2504-2289/9/11/280)  
> 4. GitHub \- EmotionLab/EEGain, [https\://github.com/EmotionLab/EEGain](https://github.com/EmotionLab/EEGain)  
> 5. Challenge \| EmoRec EEG, [https\://emotionlab.github.io/EmoRec-EEG/challenge/](https://emotionlab.github.io/EmoRec-EEG/challenge/)  
> 6. A review: Music-emotion recognition and analysis based on EEG, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC9640432/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9640432/)  
> 7. Decoding the Neural Signatures of Valence and Arousal ... \- bioRxiv, [https\://www\.biorxiv.org/content/10.1101/2021.07.23.453533v4.full-text](https://www.biorxiv.org/content/10.1101/2021.07.23.453533v4.full-text)  
> 8. Decoding the neural signatures of valence and arousal from ... \- PMC, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC9764010/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9764010/)  
> 9. Hemispheric Relation-Aware Temporal Modeling for Limited ... \- MDPI, [https\://www\.mdpi.com/2076-3417/16/14/6899](https://www.mdpi.com/2076-3417/16/14/6899)  
> 10. AffectNet: A Database for Facial Expression, Valence, and Arousal, [http\://mohammadmahoor.com/wp-content/uploads/2017/08/AffectNet\_oneColumn-2.pdf](http://mohammadmahoor.com/wp-content/uploads/2017/08/AffectNet_oneColumn-2.pdf)  
> 11. Daily Papers \- Hugging Face, [https\://huggingface.co/papers?q=neural%20prediction](https://huggingface.co/papers?q=neural+prediction)  
> 12. 935963004/LaBraM: \[ICLR 2024 spotlight\] Large Brain Model for, [https\://github.com/935963004/labram](https://github.com/935963004/labram)  
> 13. eugenehp/cbramod-rs: CBraMod EEG Foundation Model ... \- GitHub, [https\://github.com/eugenehp/cbramod-rs](https://github.com/eugenehp/cbramod-rs)  
> 14. \[ICLR 2025\] CBraMod: A Criss-Cross Brain Foundation ... \- GitHub, [https\://github.com/wjq-learning/cbramod](https://github.com/wjq-learning/cbramod)  
> 15. Naturalistic Music EEG Dataset \- Tempo (NMED-T) \- PURLs, [https\://purl.stanford.edu/jn859kj8079](https://purl.stanford.edu/jn859kj8079)  
> 16. Naturalistic Music EEG Dataset \- Tempo (NMED-T) \- Spotlight Exhibits, [https\://exhibits.stanford.edu/data/catalog/jn859kj8079](https://exhibits.stanford.edu/data/catalog/jn859kj8079)  
> 17. NMED-T: A TEMPO-FOCUSED DATASET OF CORTICAL AND, [https\://archives.ismir.net/ismir2017/paper/000198.pdf](https://archives.ismir.net/ismir2017/paper/000198.pdf)  
> 18. stabilityai/stable-audio-open-1.0 \- Hugging Face, [https\://huggingface.co/stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)  
> 19. (PDF) Stable Audio Open \- ResearchGate, [https\://www\.researchgate.net/publication/382445394\_Stable\_Audio\_Open](https://www.researchgate.net/publication/382445394_Stable_Audio_Open)  
> 20. m-a-p/MERT-v1-95M \- Hugging Face, [https\://huggingface.co/m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)  
> 21. xycld/music-align-mert \- Hugging Face, [https\://huggingface.co/xycld/music-align-mert](https://huggingface.co/xycld/music-align-mert)  
> 22. Understanding the Text and Data Mining Exception to Copyright Laws, [https\://store.lawnet.com/blog/post/understanding-the-text-and-data-mining-exception-to-copyright-laws-implications-for-training-large-language-models.html](https://store.lawnet.com/blog/post/understanding-the-text-and-data-mining-exception-to-copyright-laws-implications-for-training-large-language-models.html)  
> 23. Japan amends its copyright legislation to meet future demands in AI, [https\://eare.eu/japan-amends-its-copyright-legislation-to-meet-future-demands-in-ai-and-big-data/](https://eare.eu/japan-amends-its-copyright-legislation-to-meet-future-demands-in-ai-and-big-data/)  
> 24. Decoding the neural signatures of valence and arousal ... \- Frontiers, [https\://www\.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1051463/full](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1051463/full)  
> 25. Predicting Exact Valence and Arousal Values from EEG \- PMC \- NIH, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC8155937/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8155937/)  
> 26. EEG-based emotion recognition using graph convolutional neural, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC11294218/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294218/)  
> 27. (PDF) TSception: Capturing Temporal Dynamics and Spatial, [https\://www\.researchgate.net/publication/360127722\_TSception\_Capturing\_Temporal\_Dynamics\_and\_Spatial\_Asymmetry\_from\_EEG\_for\_Emotion\_Recognition](https://www.researchgate.net/publication/360127722_TSception_Capturing_Temporal_Dynamics_and_Spatial_Asymmetry_from_EEG_for_Emotion_Recognition)  
> 28. LGGNet: Learning from Local-Global-Graph Representations ... \- arXiv, [https\://arxiv.org/html/2105.02786v3](https://arxiv.org/html/2105.02786v3)  
> 29. (PDF) LGGNet: Learning From Local-Global-Graph Representations, [https\://www\.researchgate.net/publication/367382281\_LGGNet\_Learning\_From\_Local-Global-Graph\_Representations\_for\_Brain-Computer\_Interface](https://www.researchgate.net/publication/367382281_LGGNet_Learning_From_Local-Global-Graph_Representations_for_Brain-Computer_Interface)  
> 30. Predicting Exact Valence and Arousal Values from EEG, [https\://pdfs.semanticscholar.org/36f8/d1d568f57e62a6ed3b16e31ce232f68b2fbe.pdf](https://pdfs.semanticscholar.org/36f8/d1d568f57e62a6ed3b16e31ce232f68b2fbe.pdf)  
> 31. Hybrid CNN–LSTM Model For Continuous Stress Quantification Via, [https\://xlescience.org/index.php/IJASIS/article/download/1056/399/2313](https://xlescience.org/index.php/IJASIS/article/download/1056/399/2313)  
> 32. survey\_prompt.txt  
> 33. A Continual Learning and Context-Aware EEG Emotion Recognition, [https\://www\.preprints.org/manuscript/202505.0072](https://www.preprints.org/manuscript/202505.0072)  
> 34. TSCL-LwF: A Cross-Subject Emotion Recognition Model via Multi, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC12838903/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12838903/)  
> 35. BIOT: Cross-data Biosignal Learning in the Wild \- arXiv, [https\://arxiv.org/html/2305.10351v1](https://arxiv.org/html/2305.10351v1)  
> 36. BIOT: Biosignal Transformer for Cross-data Learning in the Wild, [https\://proceedings.neurips.cc/paper\_files/paper/2023/file/f6b30f3e2dd9cb53bbf2024402d02295-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/f6b30f3e2dd9cb53bbf2024402d02295-Paper-Conference.pdf)  
> 37. Music emotion recognition with cEEGrid \- Marcus Pearce, [https\://www\.marcus-pearce.com/assets/papers/WinnardEtAl2026.pdf](https://www.marcus-pearce.com/assets/papers/WinnardEtAl2026.pdf)  
> 38. Expectation and Acoustic Neural Network Representations Enhance, [https\://arxiv.org/html/2603.03190v1](https://arxiv.org/html/2603.03190v1)  
> 39. (PDF) Predicting artificial neural network representations to learn, [https\://www\.researchgate.net/publication/392203652\_Predicting\_artificial\_neural\_network\_representations\_to\_learn\_recognition\_model\_for\_music\_identification\_from\_brain\_recordings](https://www.researchgate.net/publication/392203652_Predicting_artificial_neural_network_representations_to_learn_recognition_model_for_music_identification_from_brain_recordings)  
> 40. CORRELATION OF EEG RESPONSES REFLECTS STRUCTURAL, [https\://zenodo.org/record/10265273/files/000030.pdf](https://zenodo.org/record/10265273/files/000030.pdf)  
> 41. 1000 songs for emotional analysis of music \| Request PDF, [https\://www\.researchgate.net/publication/262220011\_1000\_songs\_for\_emotional\_analysis\_of\_music](https://www.researchgate.net/publication/262220011_1000_songs_for_emotional_analysis_of_music)  
> 42. Computer Science \- arXiv, [https\://www\.arxiv.org/list/cs/new?skip=450\&show=1000](https://www.arxiv.org/list/cs/new?skip=450&show=1000)  
> 43. A Multimodal Framework for Continuous Music Emotion Recognition, [https\://www\.researchgate.net/publication/404777847\_A\_Multimodal\_Framework\_for\_Continuous\_Music\_Emotion\_Recognition](https://www.researchgate.net/publication/404777847_A_Multimodal_Framework_for_Continuous_Music_Emotion_Recognition)  
> 44. stabilityai/stable-audio-open-1.0 · Add details on fine-tuning, [https\://huggingface.co/stabilityai/stable-audio-open-1.0/discussions/11](https://huggingface.co/stabilityai/stable-audio-open-1.0/discussions/11)  
> 45. microsoft/fadtk: A simple library for Fréchet Audio Distance (FAD, [https\://github.com/microsoft/fadtk](https://github.com/microsoft/fadtk)  
> 46. The Flexible Copyright Exception for 'Non-Enjoyment' Purposes, [https\://www\.researchgate.net/publication/348560766\_The\_Flexible\_Copyright\_Exception\_for\_'Non-Enjoyment'\_Purposes\_-\_Recent\_Amendment\_in\_Japan\_and\_Its\_Implication](https://www.researchgate.net/publication/348560766_The_Flexible_Copyright_Exception_for_'Non-Enjoyment'_Purposes_-_Recent_Amendment_in_Japan_and_Its_Implication)  
> 47. “General Understanding on AI and Copyright in Japan”, [https\://www\.bunka.go.jp/english/policy/copyright/pdf/94055801\_01.pdf](https://www.bunka.go.jp/english/policy/copyright/pdf/94055801_01.pdf)  
> 48. The Copyright Act revised in 2018 will further improve the machine, [https\://www\.nakapat.gr.jp/en/legal\_updates\_eng/the-copyright-act-revised-in-2018-will-further-improve-the-machine-learning-environment-in-japan/](https://www.nakapat.gr.jp/en/legal_updates_eng/the-copyright-act-revised-in-2018-will-further-improve-the-machine-learning-environment-in-japan/)  
> 49. FLUX that Plays Music \- arXiv, [https\://arxiv.org/html/2409.00587v1](https://arxiv.org/html/2409.00587v1)  
> 50. braindecode/CBraMod \- Hugging Face, [https\://huggingface.co/braindecode/CBraMod](https://huggingface.co/braindecode/CBraMod)  
> 51. README.md · braindecode/Labram at main \- Hugging Face, [https\://huggingface.co/braindecode/Labram/blob/main/README.md](https://huggingface.co/braindecode/Labram/blob/main/README.md)  
> 52. labram.py \- braindecode \- GitHub, [https\://github.com/braindecode/braindecode//blob/master/braindecode/models/labram.py](https://github.com/braindecode/braindecode//blob/master/braindecode/models/labram.py)  
> 53. braindecode.models package, [https\://braindecode.org/1.0/source/braindecode.models.html](https://braindecode.org/1.0/source/braindecode.models.html)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKIAAAAVCAYAAADFPTXWAAADT0lEQVR4Xu2aW8gNURTHl1vINZciSZLkFikvnkQeXJJbHiUPbnnxuSTCkweJ5MHrx4MHl0iKeFSkUCi3BwoRcsk9ua5/e22zZjVnZpwz53xzvrN/9W/W/u9zpm/mW2fvPXsNUaAWlrKusCbajkCgURxnDZL4D+uA6gsEGgaS76DEO6QdCDScI6ypEq+l5ETsztrG6mk7yE3nk0VDTF8l+rAmUfS9QCDGS9Yr47WzVkiMJO0r8U1pez1h7Za+NL6zvlH0vaesMbFPBFqawaxF1mS2qvgz6xi5BOqq/KRRNAkkuibv9wItApLqhcRjlb9ZxQCJM1eO1s+D/lw30w4E6J6Kr6r4k4pHU5Q4mJY9U1iPVXsgawu50XMDRVM50Il3mly/5wJF/euUH2gRflB8raeTBfEEVi+Juyi/N2skxZMVbJTjL9Y03cG8k+Ma1kXl+/N+MMdAgHqQSzjsMSZtdM9gDbCmAomYxBxrCAtYSyTulFM2LnyYNQOZnGcdtWYOMOW2k3saXmb60sCoeIt1g7Xa9DU1+LV+YS0kd4F2Cgmk84bV35p1BEsE8DXmNjl48rPDO9r7jJcXu4bKki+ZBfKDNeMZitaLnQIkAy7Ker+N19HYBA4qn6oG6wucAGUpDbyfxgsE6gae1mwmzxfvpPGxXzXbeGUFpbe91iyIfqyHVMy9GG6NkjHOGuRKjM/IVY0KAwlnn/bsMIsYhXcfZ7Gdtes/VDR+X62NdUl3FIS/B2/JbctUy11rlBTUuzX++rHvafuqYjm5k55T3nrxhkob2wN3ou6m4gRV3ocDGNlqIetHOdMaiufWKDGoqW+yJrlc2WPNarhP7mZiB38l67C09XSB9izVbhYwgqMMlkZakuYBS5g0cH8rUchI0kCSfnS6VFkTegrG+25J64GkP6BMjKLoPT39rt4hOaZN/XghIYvxFJ1bV0z8tJp0zzxpibjfGgKup56MMG2UHvOAZwm9VXRZjoWsw5Fkdn1owVO1f2jBP+O96utI8PIA/n6sB3FEZeKR9K1iXWfdpqiWm8Q8ayh2kjsvRi5//mvSh6UKzv9A2pVIS8Tp1mBek5uZPJXKfrWgBxa8aX5KtdNA5WixxB/JXT8KH2f/faIK/gK0GSorbvNzbwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAUCAYAAABbLMdoAAAAiUlEQVR4XmNgGAUDCdyB+D8Q/wBiJTQ5FHALiE9B2aoMEE1sCGkEAEnMwyL2CE2MoQAqgQwYoWL30MTBguiKM6FirWjiYMHJWMT+oYkxxEIlliKJtUDFuJDEwOAuVOITEEcAsTOUz4usCAaQrdMH4lAkORQgwQBRHIYugQ2APIUeCjjBLwYiFQMA4zoi8Kbrry4AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAWCAYAAAD0OH0aAAAAlElEQVR4XmNgGAXDAswB4l9oYlJAvB1NDA7+A/EXNLHvUHEMIMcAkbBDEweJtaKJgcF8BkyTPKBi/GjiYACSQNdwAosYHIAkTmIRg2kwRZaQh0rMQBJjhYothPJRbJoHFXiLJAYzPYcBErSLkeTgkp+Q2CBQAGWjBzVY8Di6IC4gwQDR4IkugQtMYkDzECHwk4EEDQDnoSvvLwZ8dAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAWCAYAAAD0OH0aAAAAbElEQVR4XmNgGAXDBiQC8TkgvgbE15HwUWRFICAFxP/xYJABcJAJFWRCEouDimEFIInFaGJ+UHEMkM4AkeBBE18HFccAuCRAYj/QBUGgnAG3BgF0QRgASQoh8c8CsRMSHwOATEIOwnhU6eEFANlDIcOx1TztAAAAAElFTkSuQmCC>