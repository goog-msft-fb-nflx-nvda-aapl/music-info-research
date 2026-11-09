"""Builds the 'weekly update' html block of the project page from results/weekly_tables.json and the demo clips in docs/era_market/audio_dose. Imported by 92_make_page.py (function weekly_html)."""
import json, os
def weekly_html():
    T = json.load(open("results/weekly_tables.json")); D = T["doses"]; ms = T["mert_seeds"]; s0 = T["seed0"]
    rows = "".join(f"<tr><td>{c[1:]}</td><td>{ms['adapter'][c]['mean']:.2f} ± {ms['adapter'][c]['sd']:.2f}</td><td>{ms['control'][c]['mean']:.2f} ± {ms['control'][c]['sd']:.2f}</td><td>{s0['adapter'][c]['effnet']:.2f}</td><td>{s0['control'][c]['effnet']:.2f}</td></tr>" for c in D)
    def players(folder, doses):
        return "".join(f"<div class='ap'><span>dose {d[1:]}</span><audio controls preload='none' src='audio_dose/{folder}/A_007e1d02fd77_{d}_s0.ogg'></audio></div>" for d in doses)
    ds = ["a0", "a0.2", "a0.4", "a0.6", "a0.8", "a1"]
    hero = "".join(f"<div class='ap'><span>{e}</span><audio controls preload='none' src='audio_dose/hero/{c}_{e}_s0.ogg'></audio></div>" for c in ["A_01321d0221ab"] for e in ["neutral", "1960s", "1980s", "2010s"])
    return f"""<h2>Weekly update, 9 November 2026: era and market as two controls</h2>
<p>The goal is one control plane: a fixed song is regenerated as the point (release era, release market) moves. This week's results: the <b>era axis is a continuous dose</b> a (a = 0 is the 1960s prototype, a = 1 the 2010s prototype); the <b>market axis is carried by the sung language</b> (translated lyrics, vocal replaced by the target-language voice); both axes combine in one grid. All read-outs below are existing metrics; no new metric is proposed.</p>
<h3>Era dose response, adapter versus a control conditioner</h3>
<p>Expected decade index of the generated accompaniment (0 = 1960s, 5 = 2010s) per dose. MERT: mean ± sd over three training seeds (72 clips per dose and seed). Discogs-EffNet (independent of the probe encoders): one training seed. The control uses the same conditioning route without the adapter's bias injection.</p>
<div class="t"><table><tr><th>dose a</th><th>MERT adapter</th><th>MERT control</th><th>EffNet adapter</th><th>EffNet control</th></tr>{rows}</table></div>
<ul>
<li>Both curves are monotone (Spearman +1.00 for every run) and meet at the end points; chroma cosine to the source stays .81 to .84.</li>
<li>Between doses .3 and .7 the adapter is ahead of the control on MERT and EffNet in all three training seeds (dose .5: 3.25 vs 2.15 on MERT, three-seed means).</li>
<li>The distribution-level metric disagrees: FAD with sample-size extrapolation to the real 2010s does not rank the two models consistently across seeds, so we do not claim that the adapter generates more realistic audio of the decade; we claim it reaches the decade at a lower dose.</li>
<li>Beyond the end points the response saturates; a path 1960s → 1980s → 2010s increases gradually, with the 1990s and 2000s region reached by intermediate doses.</li>
</ul>
<h3>Market axis and the combined grid</h3>
<ul>
<li>The market move is carried by the sung language: erasing the language direction from the vocal probe removes it. Language share (0 to 100 % of code-switched lyrics) moves the Whisper language probability by +.71 to +.78 and the vocal market probe by +.28 (Spanish) and +.32 (German).</li>
<li>In the 5 × 5 grid (five era doses × five language shares, Spanish and German, 16 songs, three seeds) the era effect is +.62 to +1.00 expected decades at every language share and the language effect +.71 to +.78 at every era dose; the confidence intervals exclude zero.</li>
<li>Vocal-containing mixes are derived from commercial recordings and are not published here; the clips below are accompaniment only.</li>
</ul>
<h3>Evaluation protocol</h3>
<p>Controllability: probe index and gain, R-Precision and multimodal distance in embedding space (adapted from Guo et al., CVPR 2022), dose monotonicity. Distribution: FAD per cell against real clips of the decade (MERT, MuQ), extrapolated to infinite sample size (Gui et al., arXiv 2311.01616). Variety: Diversity and MultiModality. Content: song identification, chroma similarity. Baselines are re-run on the same 24 songs; numbers from other papers are not mixed in. No listening test has been run.</p>
<h3>Listen: era dose, same song and seed</h3>
<p>Generated accompaniment only, seed 0, same chroma input. Doses from 1960s-like (0) to 2010s-like (1), adapter versus control.</p>
<div class="dc"><div><b>Adapter</b>{players('adapter', ds)}</div><div><b>Control</b>{players('control', ds)}</div></div>
<p>Another song, era prompts instead of doses:</p><div class="dc"><div>{hero}</div></div>
"""
