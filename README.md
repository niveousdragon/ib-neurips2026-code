# Information-plane dynamics during learning: analysis code

Code for the NeurIPS 2026 paper *Information bottleneck dynamics during learning
across artificial and biological neural systems* (Pospelov, Ivashkina, Plusnin,
Rogozhnikova, Ivanova, Toropova, Anokhin).

The paper tracks information-plane trajectories (input retention against target
relevance) during learning in three systems:

1. a spiking ResNet-18 trained on CIFAR-10 (`snn/`);
2. mouse visual cortex during multi-week category learning, on the public dataset
   of Goltstein et al. (2021) (`cortex/`);
3. mouse hippocampus CA1 during multi-day place learning in an open field, our own
   miniscope recordings (`hippocampus/`).

The three analyses share one scaffold (`src/ib/`): an autoencoder probe of the
population activity with a supervised head, the two information-plane axes,
held-out cross-validation and, for the biological systems, a shuffle null with
shuffle-subtraction bias correction.

## Layout

```
src/ib/          shared package: autoencoder probes (estimator.py, ae_predict.py),
                 MI estimators (mi_estimators.py), data/result locations (paths.py)
snn/             01-11 raw activity -> caches; 20-24, paper_numbers.py -> numbers; fig_* -> figures
cortex/          01-03 raw data -> per-session results; 04-09 -> numbers and tables; fig_* -> figures
hippocampus/     01-10 raw data -> per-session tables; stats_* -> numbers; fig_* -> figures;
                 legacy_* the earlier 20 Hz pipeline behind two SI items; extra_* not in the paper
figures/         make_scheme.py (Table 1 schematic); rendered figures go to figures/out/
results/         small cached intermediate results (about 6 MB) that the paper numbers come from
tools/           check_release.py (scan for machine paths and credentials)
tests/           tests of the shared package
```

## Installation

Python 3.10 or newer (checked with 3.13 on Windows, CPU only).

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
pytest                            # tests of the shared package
```

`requirements.txt` pins the exact environment the release was checked in. The only
analysis dependency outside the scientific Python stack is
[DRIADA](https://pypi.org/project/driada/) 1.2.2 (autoencoder and KSG / Gaussian-copula
MI estimators). A GPU is optional; the paper runs used one NVIDIA RTX 4090.

## Where the paths come from

Scripts never contain machine paths. Inputs and outputs are resolved by `src/ib/paths.py`:

| Variable | Default | Content |
|---|---|---|
| `IB_DATA_SNN` | `data/snn` | SNN activity, labels, MEI results |
| `IB_DATA_CORTEX` | `data/cortex` | `data/chronicrecordings` of the Goltstein et al. deposit |
| `IB_DATA_HIPPOCAMPUS` | `data/hippocampus` | hippocampal session files and cell matching |
| `IB_RESULTS` | `results` | cached results, one subfolder per system |
| `IB_FIGURES` | `figures/out` | rendered figures |

## Data

**Visual cortex.** Goltstein, Reinert, Bonhoeffer and Hübener (2021), *Nature Neuroscience*
24:1441-1451, doi:10.1038/s41593-021-00914-5. Data:
<https://gin.g-node.org/pgoltstein/category-learning-visual-areas> (CC BY-SA 4.0; about 98 GB
in total). Only the `TrainingImaging` files of mice F02, F03, F04, K01 and K02 under
`data/chronicrecordings/<AREA>/<MOUSE>/` are used (95 files, about 23 GB):

```bash
gin login
gin get pgoltstein/category-learning-visual-areas
cd category-learning-visual-areas
gin get-content data/chronicrecordings     # or only the five mice's folders
export IB_DATA_CORTEX=$PWD/data/chronicrecordings
```

**Hippocampus.** Our miniscope recordings of 16 mice over 4 days (64 sessions; 61 analysed).
Data availability: TODO (DOI of the data deposit). Expected layout under `IB_DATA_HIPPOCAMPUS`:

```
sessions/NOF_<mouse>_<day>D syn data.npz    64 session files (calcium, position, behaviour)
matching/NOF_<mouse>.csv                    16 cross-day cell-matching tables
```

**Spiking network.** Spike-count activity of the trained network at 26 checkpoints on CIFAR-10
(about 85 GB) and the per-neuron maximally exciting inputs. Availability: TODO (URL or DOI).
Layout under `IB_DATA_SNN` is documented in `snn/snn_common.py`. The training of the network
itself is not part of this repository.

Without the raw data, every number and figure of the paper can still be regenerated from
`results/` (next section).

## Reproducing the paper from the cached results (minutes, CPU)

Run from the repository root. Each `stats`/`paper_numbers` script prints the paper value next
to the recomputed one.

| Paper item | Command |
|---|---|
| Table 1 schematic | `python figures/make_scheme.py` |
| **SNN** Figure 1 (information plane) | `python snn/fig_snn_main.py --fig 1` |
| Figure 2 (per-neuron MI vs MEI sharpness) | `python snn/fig_snn_main.py --fig 2` |
| H_gauss peaks and drops (Sec. 3.1, SI peak/end table, "13 sublayers drop <= 22 bits") | `python snn/21_hgauss_peaks.py` |
| Peak drift of ID and I(Z;Y) (Sec. 3.1, SI) | `python snn/20_peak_drift.py` |
| Test accuracy, MEI correlations, epoch-700 and threshold tables, lambda = 0, dequantization | `python snn/paper_numbers.py` |
| Shuffle null (Methods, SI) | `python snn/22_shuffle_null_summary.py` |
| Spike-count (Poisson) reconstruction loss (SI) | `python snn/23_poisson_summary.py` |
| Auxiliary image statistics and SI figure | `python snn/24_image_statistics_slopes.py` |
| SI figures d_Z = 8 and L3.1 scatter | `python snn/fig_supp_dim8_iplane.py`, `python snn/fig_supp_l31_scatter.py` |
| **Cortex** control tables (lambda = 0, blocked CV, shuffles) | `python cortex/04_aggregate_validation.py` |
| Stage labels, endpoints, SI input-axis numbers, centroids (Sec. 3.2) | `python cortex/05_stage_labels_and_si.py` |
| SI decoder-behaviour tables, decoder gap (Sec. 3.2) | `python cortex/06_decoder_behaviour_tables.py` |
| Mixed models, lambda = 0, per-area numbers | `python cortex/07_mixed_models_and_controls.py` |
| SI cluster table | `python cortex/08_cluster_numbers.py` |
| Headline cohort effect, V1, remaining SI numbers | `python cortex/09_paper_numbers.py` |
| Figure 3 | `python cortex/fig_cortex_ip.py` |
| SI learning-outcome figure | `python cortex/fig_supp_cortex_outcome.py` |
| **Hippocampus** slope tests, LMM, habituation, SI coverage / decoder / sensitivity tables (Sec. 3.3) | `python hippocampus/stats_main.py` |
| Per-component input axis | `python hippocampus/stats_per_component.py` |
| Matched neurons, session exclusion, visited bins | `python hippocampus/stats_cohort_facts.py` |
| SI lambda contrast and direction grid | `python hippocampus/stats_si_cohort.py` |
| Decoder family (MLP) | `python hippocampus/stats_decoder_mlp.py` |
| Alternative probes, KSG vs GCMI | `python hippocampus/stats_alt_probes.py` |
| Target-conditioned relevance | `python hippocampus/stats_conditional_relevance.py` |
| Better and worse learners | `python hippocampus/stats_better_worse.py` |
| Per-session GLM table | `python hippocampus/stats_glm.py` |
| Raw-calcium kinematics table | `python hippocampus/stats_raw_calcium_ridge.py` |
| Figure 4 | `python hippocampus/fig_4_hippocampus.py` |
| SI figures: probes, better/worse learners, GLM heatmap | `python hippocampus/fig_si_probe_panel.py`, `fig_si_better_worse.py`, `fig_si_glm_heatmap.py` |

Figure file names: `fig1_snn_information_plane.png` and `fig2_snn_neuron_mei.png` are the
paper's Figures 1 and 2; `fig3_cortex.png` and `fig4_hippocampus.png` are Figures 3 and 4.

## Reproducing from raw data (hours, GPU recommended)

Every step writes the cache the steps above read; run with `--help` for options.

**Cortex** (a few GPU-hours; timed only for the input-axis shuffle, 27 min):

```bash
python cortex/01_run_sessions.py           # AE probe per session, 5-fold CV
python cortex/02_run_validation.py         # lambda = 0, blocked CV, shift-shuffle null (3 shuffles)
python cortex/03_input_axis_shuffle.py     # input-axis shuffle null (5 shuffles)
```

**Hippocampus** (about 1 h per regime for 64 sessions on GPU, plus about 30 min of probes):

```bash
python hippocampus/01_train_probes.py --regime lambda1
python hippocampus/01_train_probes.py --regime lambda0
python hippocampus/02_ib_plane.py --regime lambda1
python hippocampus/02_ib_plane.py --regime lambda0
python hippocampus/02_ib_plane.py --regime lambda1 --decoder mlp
python hippocampus/03_decoder_accuracy.py --regime lambda1
python hippocampus/04_behavior.py
python hippocampus/05_per_component_mi.py --regime lambda1
python hippocampus/05_per_component_mi.py --regime lambda0
python hippocampus/06_alt_probes.py
python hippocampus/07_conditional_relevance.py
python hippocampus/08_ksg_vs_gcmi.py
python hippocampus/09_glm_latents.py --regime lambda1
python hippocampus/09_glm_latents.py --regime lambda0
python hippocampus/10_session_summary.py
# earlier 20 Hz pipeline, used only for the SI GLM heatmap and the raw-calcium table:
python hippocampus/legacy_01_pilot_20hz.py --mouse <M> --day <D> [--shuffled]   # per session
python hippocampus/legacy_02_glm_r2.py
python hippocampus/legacy_03_raw_calcium_ridge.py
```

**Spiking network** (on the order of 100 GPU-hours for all sweeps):

```bash
python snn/01_dimensionality.py
python snn/02_latent_mi.py                                     # d_Z = 16, lambda = 1
python snn/02_latent_mi.py --latent-dim 8 --relevance-only --test-split-fallback --out latent_mi_dim8.npz
python snn/02_latent_mi.py --class-weight 0 --layers panel --relevance-only --out latent_mi_lambda0.npz
python snn/03_shuffle_null.py                                  # 4 h on GPU
python snn/04_unclipped_shuffle_null.py
python snn/05_poisson_reconstruction.py --epochs 900 --out poisson_recon_e900.csv
python snn/05_poisson_reconstruction.py --epochs 30 100 300 --out poisson_recon_trajectory.csv
python snn/06_dequantization_check.py
python snn/07_mei_convert.py
python snn/08_neuron_mi_mei.py
python snn/09_fetch_cifar.py
python snn/10_image_statistics_decoding.py
python snn/11_image_statistics_bootstrap.py
```

`hippocampus/extra_vae_dz_sweep.py` (variational-autoencoder probe and a sweep of the
latent size) is an additional robustness analysis; its results are not part of the paper.

## Numerical reproducibility

- Cached results were produced with a development build of DRIADA that predates 1.2.2.
  The autoencoder, KSG and Gaussian-copula code used here is unchanged in 1.2.2; one
  quantity, the head-direction component of the hippocampal per-component analysis, comes
  out slightly different under 1.2.2 (for example 0.066 instead of 0.061 bits in one session).
- Autoencoder training on GPU and CPU is not bit-identical, and the SNN autoencoders were
  trained without a fixed seed, so a rerun from raw data reproduces the cached values up to
  training noise.
- Figures rendered with a different FreeType build differ from the paper's PNGs in text
  anti-aliasing only.

## License and citation

Code: MIT (see `LICENSE`). The Goltstein et al. data are distributed under CC BY-SA 4.0 by
their authors. If you use this code, please cite the paper (`CITATION.cff`).
