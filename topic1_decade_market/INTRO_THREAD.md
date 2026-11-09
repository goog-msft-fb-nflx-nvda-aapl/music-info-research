# Time Traveler: Two-Axis Era and Market Control for Cover Song Generation

**Team:** James Christian, Liang-Yu Hsia, Jefferson Sy Dionisio

**Idea.** One control plane (release era × release market). Keep a song's melody fixed and move the point on the plane: the cover is regenerated as if it came from another decade and another market, for example a recent US song as a Brazilian 2010 recording.

**How.** Era is a continuous dose on a frozen generator (MusicGen-Melody, ACE-Step for the market route); market is carried by the sung language (translated lyrics, target-language voice). Evaluation uses existing metrics only (probe index, FAD, R-Precision, Diversity, MultiModality, song identification).

**Status (9 Nov 2026).**
- Era dose response is monotone across three training seeds; the adapter reaches the target decade at a lower dose than a control conditioner (MERT index at dose .5: 3.25 vs 2.15), while FAD does not rank the two consistently.
- Market and era effects combine in a 5 × 5 grid (Spanish, German): era +.62 to +1.00 expected decades, language +.71 to +.78, intervals exclude zero.
- No listening test; probes are proxies.

**Links.** Page and audio demos: https://goog-msft-fb-nflx-nvda-aapl.github.io/music-info-research/era_market/ · Weekly video: https://youtu.be/DvFWE80UYQ8 · Code and results: https://github.com/goog-msft-fb-nflx-nvda-aapl/music-info-research (folder `topic1_decade_market`)

**Looking for:** pairs of different recordings of the same song from different decades or markets (Discogs-VI metadata), and feedback on the evaluation protocol.
