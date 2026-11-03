# EEG Emotion Dataset Access Audit — 3 November 2026

## Executive finding

The access situation is materially different across the requested datasets. **SEED / SEED-IV / SEED-V is the only one of the five legacy families whose official application workflow I could verify as live today.** The official BCMI landing page, application form, licence page, and contact address are reachable. citeturn352001view4turn652370search0turn990788view0turn857443search1

For **DEAP**, the user-facing dataset landing page timed out when opened directly today; the official EULA PDF is still retrievable/indexed, but the application workflow itself could not be verified. SciCrunch now labels the resource “THIS RESOURCE IS NO LONGER IN SERVICE,” last updated 31 October 2026, and an independent 2026 dataset-access audit reported that DEAP's documented route was non-functional on 11 September 2026. citeturn352001view0turn854339view0turn883921search3turn883921search2

For **AMIGOS**, the official landing page returned **502 Bad Gateway** today. The EULA remains retrievable, but its application URL points back to the unavailable QMUL site. A 2026 Frontiers paper gives a QMUL repository URL, but that URL resolves to the **accepted paper**, not the dataset files, so it is not an authorized data mirror. citeturn352001view1turn934302search5turn738605search0turn738605search9

For **MAHNOB-HCI**, `https://mahnob-db.eu/` itself is reachable today, but clicking its HCI-Tagging link produced **404 Not Found**. The EULA PDF is still reachable and confirms the old account-registration workflow; that registration route therefore remains **UNVERIFIED as operational**. citeturn352001view2turn197789view0turn182377view0

For **DREAMER**, the canonical Zenodo record is still indexed by Zenodo as a restricted dataset with an active-looking “Request access” workflow, but my direct fetch of the record timed out today. I therefore treat actual submission/download functionality as **UNVERIFIED**, not as working. citeturn322728search0turn352001view3

The strongest alternative for a student team that needs **music-listening EEG plus valence/arousal without a signed agreement** is no longer DEAP: there are now genuinely open datasets on OpenNeuro/NEMAR and Zenodo, including **ds002721**, **MET/ds008701**, and **DAAMEE**. citeturn715413view4turn258298search4turn715413view0

## Current access table

| Dataset | Status today | Official URL | Authorized mirror URL | Access steps | Who can apply | Licence / terms | Size / format | Verified |
|---|---|---|---|---|---|---|---|---|
| **DEAP** | **PAGE DOWN / UNVERIFIED application workflow.** Direct open timed out today. citeturn352001view0 | `https://www.eecs.qmul.ac.uk/mmv/datasets/deap/` | **None found that I can verify as owner-authorized.** Do not use random GitHub/HF/Kaggle copies. | Legacy procedure: permanent academic/research staff member signs EULA; up to five same-institution researchers can be named; institutional email + institutional webpage required. Official EULA says: **“The user may only use the dataset after this EULA has been signed and returned”**. citeturn854339view0 | Students can be included under a permanent-position academic signer; the EULA does **not** make the student the responsible signatory. citeturn854339view0 | Academic research only; no commercial use; no redistribution except limited publication illustrations with participant consent; citation required. citeturn854339view0 | Official current package sizes: **UNVERIFIED** because download page is unavailable. Historically raw BDF and preprocessed MATLAB/Python variants exist, but exact current byte sizes cannot be verified today. | **No** for access; **yes** for EULA |
| **DREAMER** | **RESTRICTED; direct functionality UNVERIFIED today.** Zenodo still indexes request-access form, but direct page fetch timed out. citeturn322728search0turn352001view3 | `https://zenodo.org/records/546113` | No second owner-authorized dataset host found. Durham and UWS host the **paper**, not DREAMER data. citeturn322728search1turn322728search2 | Zenodo request form asks for name, email/contact details, institution, position and reason. Exact condition: **“Please state your name, contact details (e-mail), institution, position, as well as the reason”**. citeturn322728search0 | Students are not excluded by the published condition; supervisor signature is **not stated**. Institutional affiliation is requested. citeturn322728search0 | Dataset-specific licence text is not exposed in the fetched record excerpt: **UNVERIFIED**. Citation to Katsigiannis & Ramzan is explicitly requested. citeturn322728search0 | Single MATLAB dataset is commonly referenced as `DREAMER.mat`, but exact current file size is **UNVERIFIED** because Zenodo hides restricted-file details. citeturn553780search5turn553780search6 | **Partial / UNVERIFIED download** |
| **AMIGOS** | **PAGE DOWN.** Direct request returned **502 Bad Gateway**. citeturn352001view1 | `https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/` | **No authorized dataset mirror verified.** `https://qmro.qmul.ac.uk/xmlui/handle/123456789/86242` is the accepted paper, not data. HF copy below is not demonstrably authorized. citeturn738605search9turn738605search4turn738605search7 | Permanent academic/research staff member signs; up to five researchers may be added; institutional email and institutional webpage required; signed EULA uploaded via the dataset site. Exact wording: **“This document must be signed by a person with a permanent position at an academic/research institute”**. citeturn934302search5 | Student team can be listed under a permanent academic/research staff signer. citeturn934302search5 | Academic only; commercial use prohibited; redistribution prohibited except small publication/presentation portions subject to consent; citation required. citeturn934302search5 | Preprocessed MATLAB files are `Data_Preprocessed_P01.mat` … `P40.mat`; an HF re-upload totals **3.88 GB**, but because that copy is not proven authorized, treat that only as evidence of approximate preprocessed content, not a legal source. citeturn934302search0turn738605search4 | **No** |
| **MAHNOB-HCI** | Main domain works; HCI-Tagging URL returns **404**. Registration therefore **UNVERIFIED**. citeturn352001view2turn197789view0 | `https://mahnob-db.eu/` and legacy `https://mahnob-db.eu/hci-tagging/` | No owner-authorized mirror verified. | EULA requires permanent academic signer and allows up to five same-institution researchers; legacy registration URL is `http://mahnob-db.eu/hci-tagging/accounts/register/`. Exact wording: **“This document must be signed by a person with a permanent position at an academic institute”**. citeturn182377view0 | Students may be named as additional researchers under the responsible permanent academic signer. citeturn182377view0 | Academic research only; no redistribution; publication use only where participant permission permits it; acknowledgement and paper citation required. citeturn182377view0turn978778view1 | Physiological recordings are BioSemi **BDF**, including 32 EEG channels; current downloadable total size **UNVERIFIED** because the HCI route is broken. citeturn790387search4 | **Partial: EULA yes; access no** |
| **SEED** | **WORKING application workflow.** citeturn652370search0turn990788view0 | `https://bcmi.sjtu.edu.cn/home/seed/` | No owner-authorized public mirror located. | Download licence, complete it, upload PDF at application form, verify email, await review and download link/password. citeturn990788view0turn652370search0 | Application form explicitly supports students and asks for **Supervisor Name** and **Supervisor Email**; institutional email requested. citeturn652370search0 | Academic only; redistribution prohibited; citation required. Exact wording from current SEED-V licence: **“The user may not distribute the dataset or portions thereof in any way”**. citeturn854339view1 | Raw EEG and extracted-feature MAT files exist; exact current download size **UNVERIFIED** from official site. | **Yes** |
| **SEED-IV** | **WORKING via same application form.** Description page reachable today. citeturn990788view2turn652370search0 | `https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html` | No authorized open mirror verified. | Same SEED application; select SEED-IV in checklist. citeturn652370search0 | Students accepted through application form with supervisor details. citeturn652370search0 | Same SEED-family academic/no-redistribution framework. citeturn990788view0turn854339view1 | Official page documents raw EEG and processed PSD/DE features; 62 channels, downsampled to 200 Hz during processing. Exact archive size **UNVERIFIED**. citeturn990788view2 | **Yes** |
| **SEED-V** | **WORKING via same application form.** Description page reachable. citeturn990788view3turn652370search0 | `https://bcmi.sjtu.edu.cn/home/seed/seed-v.html` | No authorized open mirror verified. | Same application; select SEED-V. citeturn652370search0 | Current form supports students with supervisor information; older SEED-V PDF says PI/director information should be used instead of student details, so the **current web form should govern the submission**. citeturn652370search0turn854339view1 | Academic only; no redistribution; mandatory citation. citeturn854339view1 | EEG + eye movement; 62-channel EEG. Exact archive size **UNVERIFIED**. citeturn990788view3 | **Yes** |
| **BCMI-MIdAS / OpenNeuro ds002721** | **DOWNLOADABLE NOW**, no agreement. citeturn715413view4 | `https://openneuro.org/datasets/ds002721` | Authorized NEMAR copy: `https://www.nemar.org/dataset/on002721` — NEMAR explicitly identifies it as a copy of OpenNeuro ds002721. citeturn715413view4 | Click zip or use `nemar dataset download on002721`; no application. citeturn715413view4 | Anyone. | **CC0**. citeturn715413view4 | **3.35 GB / 3.4-GB zip**, BIDS EEG; 31 participants, raw/unprocessed EEG at 1 kHz, film-score music and self-reported emotional responses. citeturn715413view4 | **Yes** |
| **MET / OpenNeuro ds008701** | **DOWNLOADABLE NOW**, no signed agreement. citeturn258298search4turn891800search3 | `https://openneuro.org/datasets/ds008701` | NEMAR: `https://www.nemar.org/dataset/on008701` citeturn891800search3 | Direct OpenNeuro/NEMAR download. | Anyone. | **CC0** on OpenNeuro. citeturn258298search4 | **13.06 GB**; BIDS/BrainVision; 20 participants; source recording 128 channels/1 kHz, released data preprocessed to 66 channels; valence and arousal 1–9. citeturn258298search4 | **Yes** |
| **DAAMEE** | **DOWNLOADABLE NOW** on Zenodo. citeturn715413view0 | `https://zenodo.org/records/16738766` | Not needed; Zenodo record is deposited by the named creators. | Direct per-file download; no request form. citeturn715413view0 | Anyone. | A `LICENSE` file is present, but its actual text was not parsed in my retrieval: **UNVERIFIED licence wording**. citeturn715413view0 | **32.8 GB**, including scalp EEG + cEEGrid, BIDS components, behavioural emotional responses; `sourcedata.zip` alone 28.2 GB. citeturn715413view0 | **Yes for access; licence UNVERIFIED** |
| **Valdés-Alemán et al. 2022 music EEG** | **DOWNLOADABLE NOW**. citeturn715413view1 | `https://data.mendeley.com/datasets/9fm94df3x2/1` | Mendeley is the primary deposited copy. | Direct “Download All”; no agreement. citeturn715413view1 | Anyone. | **CC BY 4.0**. citeturn715413view1 | 25 participants; valence/arousal/familiarity/pleasure. The deposited electrophysiological values are FFT-derived theta/alpha measures at F3/F4/P3/P4 rather than a full continuous raw EEG archive, so it is less suitable if you require raw signals. citeturn715413view1 | **Yes** |
| **Valdés-Alemán 2026 music–colour EEG** | **DOWNLOADABLE NOW**. citeturn715413view2 | `https://data.mendeley.com/datasets/xjtxnn7bx7/1` | Primary Mendeley deposit. | Direct download. | Anyone. | **CC BY 4.0**. citeturn715413view2 | 33 participants; valence/arousal/pleasure; CSV contains EEG spectral/asymmetry features, **not raw continuous EEG**. citeturn715413view2 | **Yes** |

## Dataset-by-dataset verification notes

### DEAP

The exact official URL tested was:

`https://www.eecs.qmul.ac.uk/mmv/datasets/deap/`

It timed out in my direct retrieval today. citeturn352001view0 The official EULA URL still resolves in indexed form:

`https://eecs.qmul.ac.uk/mmv/datasets/deap/doc/eula.pdf`

The EULA permits academic research only, prohibits commercial uses and redistribution, allows a permanent-position academic/research staff member to take responsibility for up to five same-institution researchers, requires affiliation email plus an institutional webpage, and requires citation of Koelstra et al. citeturn854339view0

The most important exact EULA phrases are **“The dataset may only be used for academic research”** and **“The user may not distribute the dataset or portions thereof in any way”**. citeturn854339view0

For publishing models, figures and numerical results: the EULA does not forbid derived analytical results, and it expressly discusses papers/presentations. Publication of participant EEG/audio-video excerpts is restricted by individual consent; ordinary aggregate model metrics and derived results are therefore consistent with the described academic-publication use. This is an interpretation of the EULA, not legal advice. citeturn854339view0

I attempted an Internet Archive URL of the form:

`https://web.archive.org/web/20261019/https://www.eecs.qmul.ac.uk/mmv/datasets/deap/`

but the archive URL was inaccessible through the browsing tool, so its contents are **UNVERIFIED**. citeturn970513view0 A separate 2026 BCI Report record says it captured the source terms on 19 October 2026, but that does not prove the live application endpoint worked. citeturn738605search7

A 2026 Frontiers appraisal independently checked DEAP on 11 September 2026 and found its documented route non-functional. SciCrunch marked DEAP “THIS RESOURCE IS NO LONGER IN SERVICE,” with a record update on 31 October 2026. citeturn883921search2turn883921search3

I did **not** find a dataset-owner-authorized Zenodo, Figshare, OSF, OpenNeuro, PhysioNet or Hugging Face copy. A current GitHub reproducibility project explicitly tells users not to redistribute DEAP and to obtain it from the provider instead. citeturn553780search5

**Contact:** a currently working DEAP-specific administrator email could not be verified from the accessible official material: **UNVERIFIED**. Do not rely on an arbitrary mirror uploader.

**Response time:** no reliable 2025–2026 owner-published turnaround time was found: **UNVERIFIED**.

### DREAMER

Canonical primary source:

`https://zenodo.org/records/546113`

The Zenodo record remains indexed as **Restricted** and says users should request access through the form. citeturn322728search0 The record names two current contact addresses:

`Stamos.Katsigiannis@durham.ac.uk`  
`Naeem.Ramzan@uws.ac.uk`

The Durham institutional repository currently lists Dr Stamos Katsigiannis as an Associate Professor and exposes his DREAMER publication record, while UWS continues to host the publication record. This supports the institutional provenance of those domains, although actual mailbox deliverability cannot be tested without sending mail. citeturn322728search1turn322728search2

The direct Zenodo page itself timed out when I opened it today, so I cannot certify that pressing “Request access” successfully completes submission on 3 November 2026. That portion is **UNVERIFIED**. citeturn352001view3

Zenodo's current general documentation confirms that restricted-record access is controlled by the record owner and that Zenodo staff cannot grant access themselves. citeturn883921search6turn883921search7

I found a Hugging Face DREAMER derivative:

`https://huggingface.co/datasets/monster-monash/DREAMERV`

but it is processed for classification and is not the original owner's distribution. I found no evidence that the uploader is Katsigiannis, Ramzan, Durham or UWS; therefore I do **not** recommend it as a legal replacement for the original files. citeturn322728search7

**Licence:** the dataset-specific licence text was not exposed in the retrieved Zenodo record, so redistribution/model-weight implications beyond the stated request condition are **UNVERIFIED**. The CC BY-NC-ND licence visible on UWS applies to the accepted **paper**, not automatically to the dataset. citeturn322728search2

**Response time / fee:** no reliable 2025–2026 evidence of a standard response time or fee was found. Both are **UNVERIFIED**.

### AMIGOS

Official URL tested:

`https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/`

Result today: **502 Bad Gateway**. citeturn352001view1

The official EULA remains indexed at:

`https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/eula.pdf`

Its access model is essentially the DEAP model: permanent-position academic/research signer, up to five additional same-institution researchers, affiliation email, institutional webpage, signed EULA upload, academic-only use, no redistribution, and citation obligation. citeturn934302search5

A 2026 paper states:

`https://qmro.qmul.ac.uk/xmlui/handle/123456789/86242`

as the AMIGOS dataset location. citeturn934302search2 I opened the QMUL repository content behind that handle and found the **accepted AMIGOS article**, not dataset files. Thus this cannot presently be treated as a data mirror. citeturn738605search9

A Hugging Face re-upload exists:

`https://huggingface.co/datasets/musicji83/amigos-eeg-dataset`

and contains 41 files totaling about **3.88 GB**. citeturn738605search4 However, BCI Report notes that the listing has no licence tag and matches a corpus whose original source requires a signed agreement. There is no evidence that the uploader is an AMIGOS author or QMUL. I therefore classify it as **authorization not established; do not recommend**. citeturn738605search7turn738605search8

A 2026 ResearchGate post reports a student receiving 404 from the official download route and no author reply; this is anecdotal rather than authoritative but is consistent with the current outage. citeturn652370search9 The stronger evidence is the 2026 Frontiers audit, which found AMIGOS unreachable through its documented route on 11 September 2026. citeturn738605search1

The Internet Archive URL I tried was:

`https://web.archive.org/web/20261019/https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/`

The browsing tool could not access the archive URL, so archive contents are **UNVERIFIED**. citeturn970513view1

**Contact:** the paper lists `j.a.mirandacorrea@qmul.ac.uk` and `i.patras@qmul.ac.uk`, but current mailbox validity is **UNVERIFIED**. citeturn857443search5

**Response time / fee:** **UNVERIFIED**; no reliable current SLA or fee was found.

### MAHNOB-HCI

Main domain:

`https://mahnob-db.eu/`

is currently reachable. citeturn352001view2 Its HCI-Tagging link leads to:

`https://mahnob-db.eu/hci-tagging/`

which returned **404 Not Found** today. citeturn197789view0

The official EULA remains retrievable:

`https://mahnob-db.eu/hci-tagging/media/doc/eula.pdf`

It requires a permanent academic-institute signer, permits up to five additional researchers, prohibits redistribution, restricts publication of subject data according to consent, and requires academic-only research. citeturn182377view0turn978778view0turn978778view1

The EULA's legacy registration URL is:

`http://mahnob-db.eu/hci-tagging/accounts/register/`

Because the parent HCI-Tagging route is broken, account registration is **UNVERIFIED as working**. citeturn182377view0

The EULA gives `laughter_db@mahnob-db.eu` for sending publications, but it does not clearly present that address as the current access-support address; current deliverability is therefore **UNVERIFIED**. citeturn182377view0

The 2026 Frontiers access audit independently reported MAHNOB-HCI's documented route non-functional on 11 September 2026. citeturn738605search1

The Internet Archive URL attempted was:

`https://web.archive.org/web/20261019/https://mahnob-db.eu/hci-tagging/`

The archive could not be fetched through the tool, so snapshot content is **UNVERIFIED**. citeturn970513view2

**Response time / fee:** **UNVERIFIED**.

### SEED, SEED-IV and SEED-V

Primary landing page:

`https://bcmi.sjtu.edu.cn/home/seed/`

is working today. citeturn352001view4

Current application form:

`https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/`

is also working and explicitly requests name, **institutional email**, country, institution, department, position, supervisor name/email for students, signed licence PDF and dataset selection. citeturn652370search0

Current licence instructions:

`https://bcmi.sjtu.edu.cn/home/seed/downloads.html`

state that applicants must complete the licence, upload it on the APPLY page, and that a download link/password will be sent after review. citeturn990788view0

Current contact:

`seed2022@sjtu.edu.cn`

is explicitly published on the BCMI SEED contact page. That proves it is the current advertised address; actual SMTP delivery was not tested. citeturn857443search1

For SEED-V, the legacy PDF gives the same academic-only/no-redistribution model and says the scanned agreement can be sent to `seed2022@sjtu.edu.cn`. It also says PI/director information should be supplied instead of student information. citeturn854339view1 The **current online form**, however, explicitly accommodates a student applicant by asking for supervisor information, so for 2026 I would follow the live form rather than the older PDF workflow. citeturn652370search0

SEED-IV page:

`https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html`

is live and documents raw EEG, 62 channels, raw-to-processed downsampling to 200 Hz, and PSD/DE feature extraction. citeturn990788view2

SEED-V page:

`https://bcmi.sjtu.edu.cn/home/seed/seed-v.html`

is live and documents 62-channel EEG plus eye movement, with five emotional categories. citeturn990788view3

No official page gives a reliable present-day turnaround time; therefore response time is **UNVERIFIED**. No application fee is stated on the live form or licence page; I found **no evidence of a fee**, but an explicit “free of charge” statement was not found, so fee status is **UNVERIFIED rather than confirmed zero**. citeturn652370search0turn990788view0

## Open music-listening EEG alternatives

The most important new finding is that several datasets now satisfy the user's “music listening + affect rating + no signed agreement” requirement much better than the legacy corpora.

**BCMI-MIdAS / ds002721** is particularly strong. It contains 31 participants listening to 40 twelve-second film-score excerpts, unprocessed EEG at 1 kHz, resting-state data and self-reported induced affect. NEMAR provides a direct 3.4-GB zip and explicitly identifies itself as a copy of OpenNeuro ds002721. The licence is **CC0**. citeturn715413view4

Official URLs:

`https://openneuro.org/datasets/ds002721`  
`https://www.nemar.org/dataset/on002721`  
`https://data.nemar.org/on002721/v1.0.0/`

NEMAR even provides the executable command:

```bash
nemar dataset download on002721
```

citeturn715413view4

The related BCMI calibration/training/testing datasets are also openly hosted. For example, ds002722 contains EEG/GSR/ECG from music clips targeting valence/arousal states and is CC0 on OpenNeuro. citeturn258298search1 The ds002724 repository states that its rights-holder is the University of Reading and licenses that dataset under **CC BY 4.0**, giving unusually clear provenance. citeturn258298search2

**MET / ds008701** is newer and larger. It contains 20 subjects, 20 songs per subject, SAM valence/arousal ratings, likeability/familiarity, and released EEG in BIDS/BrainVision format. The current OpenNeuro record reports **13.06 GB** and **CC0**. citeturn258298search4

URLs:

`https://openneuro.org/datasets/ds008701`  
`https://www.nemar.org/dataset/on008701`

NEMAR reports the downloadable archive at about 13.0 GB. citeturn891800search3

**DAAMEE**, published in 2025 by named QMUL/Aarhus researchers, is an open Zenodo deposit with scalp EEG and cEEGrid data, original musical stimuli and behavioural emotional responses. The record contains 32.8 GB total and direct download links. citeturn715413view0

URL:

`https://zenodo.org/records/16738766`

The licence file exists in the record, but I did not retrieve its contents, so its exact reuse terms are **UNVERIFIED**. citeturn715413view0

I also found two Mendeley deposits that meet the broad music/emotion criterion but are less useful for raw-EEG modelling because the downloadable electrophysiology is already reduced to spectral/asymmetry measures. The 2022 deposit is CC BY 4.0 and includes valence/arousal/familiarity/pleasure from 25 participants. citeturn715413view1 The March 2026 deposit is also CC BY 4.0 and includes valence/arousal/pleasure plus spectral EEG features from 33 participants. citeturn715413view2

URLs:

`https://data.mendeley.com/datasets/9fm94df3x2/1`  
`https://data.mendeley.com/datasets/xjtxnn7bx7/1`

I searched the requested repository classes—OpenNeuro/NEMAR, Zenodo, Mendeley Data, PhysioNet, Kaggle, Hugging Face, plus web-indexed Figshare/OSF/IEEE DataPort results. Kaggle contains multiple emotion/music redistributions, but I found no evidence making the DEAP/AMIGOS copies owner-authorized. Hugging Face similarly contains agreement-gated corpus re-uploads whose uploader rights are not established. The 2026 BCI Report independently found many HF EEG uploads whose stated licensing differed from the source corpus terms. citeturn738605search8 This search cannot prove that no additional qualifying dataset exists in every repository, so any negative repository-level finding is **UNVERIFIED/exhaustiveness not guaranteed**.

## Reports from users and researchers in 2025–2026

The most reliable 2026 evidence is the Frontiers neuroinformatics appraisal published in October 2026. Its authors rechecked access routes on 11 September 2026 and found **DEAP, AMIGOS and MAHNOB-HCI** unreachable through their documented routes. citeturn883921search2

SciCrunch independently marks DEAP “THIS RESOURCE IS NO LONGER IN SERVICE,” with its registry record updated on 31 October 2026. citeturn883921search3

A 2026 AMIGOS ResearchGate post reports the official QMUL download being broken and an author not responding. Because this is a user report rather than an authoritative status service, it should be treated as anecdotal corroboration only. citeturn652370search9

A June 2026 Frontiers AMIGOS paper claims the dataset is available from a QMUL repository handle, but inspection shows that handle hosts the accepted paper rather than the corpus. This is a useful warning against trusting recent papers' data-availability links without opening them. citeturn934302search2turn738605search9

For SEED, the official infrastructure is currently working; I did not find a reliable 2025–2026 report giving a typical approval turnaround. **UNVERIFIED response time.** citeturn652370search0turn990788view0

For DREAMER, Zenodo's own current documentation says restricted-file access remains under the record owner's control and Zenodo support cannot grant it. Therefore, even if the form submits successfully, the approval latency depends on the owners. citeturn883921search6

## Ranked recommendation for a student team in Taiwan

**Fastest: BCMI-MIdAS / OpenNeuro ds002721.** It is already downloadable, has raw music-listening EEG, affective self-report, a CC0 licence, and no agreement, institutional email or supervisor signature. citeturn715413view4 **Exact first step:** open `https://www.nemar.org/dataset/on002721` and download the 3.4-GB archive, or run:

```bash
nemar dataset download on002721
```

**Second: MET / ds008701.** It is also immediately downloadable and CC0, with explicit valence/arousal ratings and a modern BIDS structure, but at roughly 13 GB it is substantially larger. citeturn258298search4turn891800search3 **Exact first step:** open `https://openneuro.org/datasets/ds008701` and select the current version.

**Third: DAAMEE.** Immediate Zenodo download and excellent music-emotion relevance, but at 32.8 GB it is heavier, and I could not verify the licence text despite the deposited `LICENSE` file. citeturn715413view0 **Exact first step:** open `https://zenodo.org/records/16738766` and inspect/download the `LICENSE` file before downloading participant archives.

**Fourth: SEED / SEED-IV / SEED-V.** This is the best currently functioning route among the traditional gated emotion datasets. The live application form explicitly permits student applicants to enter supervisor details and requests an institutional email. citeturn652370search0 **Exact first step:** download the licence from `https://bcmi.sjtu.edu.cn/home/seed/downloads.html`, have the appropriate supervisor/PI details ready, then submit it at `https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/`.

**Fifth: DREAMER.** The record still advertises a Zenodo access-request mechanism, but direct record access timed out in my test today, so submission success and approval latency remain **UNVERIFIED**. citeturn322728search0turn352001view3 **Exact first step:** try `https://zenodo.org/records/546113`; if the request form loads, submit name, institutional contact, position and research purpose. If not, use the currently published author contacts `Stamos.Katsigiannis@durham.ac.uk` and `Naeem.Ramzan@uws.ac.uk`. citeturn322728search0

**DEAP, AMIGOS and MAHNOB-HCI should not be the first choice for a time-sensitive student project on 3 November 2026.** DEAP timed out and is now marked out of service by SciCrunch; AMIGOS returned 502; MAHNOB's HCI-Tagging path returned 404. Their old EULAs are still recoverable, but the actual access routes cannot presently be verified as operational. citeturn352001view0turn883921search3turn352001view1turn197789view0 For all three, unauthorized Hugging Face/Kaggle/cloud re-uploads should be avoided because their original terms restrict redistribution and I found no owner authorization for those copies. citeturn854339view0turn934302search5turn182377view0turn738605search7