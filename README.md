# Step 07 — Pulsicity & Flow: Kinematic Behavioral Analysis
### Gaga–Psilocybin Study | Academic Peer Review Package

**Copyright © 2026 Dror Hazan. All Rights Reserved.**
Access is granted solely for academic peer review. See [LICENSE](LICENSE) for terms.

---

## 1. Project Overview

This package contains the **Step 07** behavioral metrics pipeline for the Gaga–Psilocybin motion capture study. It operates on full-body kinematic data (41-marker OptiTrack system, 120 Hz) to characterize the temporal structure of movement — specifically, how often and how smoothly body segments generate discrete velocity pulses during free Gaga dance improvisation.

The three recordings provided are from Subject 651, spanning three experimental sessions (T1, T2, T3). All data were processed through a validated six-step kinematic pipeline before reaching this notebook.

The primary metrics computed here are:

| Metric | Symbol | Description |
|---|---|---|
| Peaks Per Minute | PPM | Rate of discrete velocity pulses, normalized to active movement time |
| Inter-Peak Interval CV | IPI CV | Temporal regularity / rhythmic consistency of pulses |
| Spectral Arc Length | SPARC | Movement smoothness in the frequency domain *(see §6 disclaimer)* |

---

## 2. Data Provenance & the Measurement Signal $v_m$

### 2.1 Root-Relative Position (Step 06, NB06)

Raw marker positions from OptiTrack are expressed in the global lab frame. In the preceding pipeline step (Notebook 06), each segment's 3D world position is made **root-relative** by subtracting the 3D position of the pelvis segment (Hips) at every frame:

$$\mathbf{p}_{rel}(t) = \mathbf{p}_{segment}(t) - \mathbf{p}_{Hips}(t)$$

This isolates the body's **internal postural configuration** from any global translation across the room, ensuring that locomotion through space does not contaminate the movement quality signal.

### 2.2 Velocity Derivation via Savitzky-Golay Filter

The 3D linear velocity components ($v_x, v_y, v_z$) are computed by **differentiating** the root-relative positions. This is done using a NaN-safe chunked Savitzky-Golay filter with `deriv=1`:

| SG Parameter | Value |
|---|---|
| Sampling rate | 120 Hz |
| Window length | 21 frames (0.175 s) |
| Polynomial order | 3 |
| Mode | `interp` (handles boundary frames) |
| Derivative order | 1 (velocity) |

This filter differentiates and smooths simultaneously in a single pass. Its standard **−3 dB effective cutoff frequency is ~6.11 Hz** — meaning velocity fluctuations above ~6 Hz are attenuated. This is the relevant spectral boundary for all downstream analysis (SPARC integration cap, secondary filter ceiling).

### 2.3 The Measurement Signal $v_m$

The column `{segment}__lin_vel_rel_mag` — referred to throughout the notebook as $v_m$ — is the **Euclidean magnitude** of the three velocity components:

$$v_m(t) = \sqrt{v_x(t)^2 + v_y(t)^2 + v_z(t)^2} \quad [\text{mm/s}]$$

**$v_m$ is the ground-truth signal. It is never re-filtered or modified by Step 07.**

> **For a full mathematical breakdown of the Step 06 coordinate frames, Euler angle conventions, artifact flagging thresholds, and the complete velocity derivation logic, please read the included [`KINEMATIC_FEATURES_README.md`](KINEMATIC_FEATURES_README.md).**

---

## 3. Required Input Data

The notebook expects three **Step 06 kinematic Parquet files** of the form:

```
{RUN_ID}__kinematics_master.parquet
```

Each file contains one row per frame (120 Hz) with all 41-segment kinematic columns including `{segment}__lin_vel_rel_mag` and `{segment}__is_artifact` flags.

> **⚠ Data files are not included in this package due to file size (~200 MB each, ~651 MB total).**
> Please download the three Parquet files from the secure cloud storage link provided separately and place them directly into the `data/` folder before running the notebook:
>
> ```
> publish4Jason/
> └── data/
>     ├── 651_T1_P1_R1_...parquet
>     ├── 651_T2_P1_R1_...parquet
>     └── 651_T3_P1_R1_...parquet
> ```

---

## 4. The Step 07 Computation

### 4.1 Dual-Signal Architecture

Step 07 operates on **two parallel signals** derived from $v_m$:

| Signal | Symbol | Role |
|---|---|---|
| Measurement Signal | $v_m$ | Ground-truth magnitudes. Never filtered. Artifact frames are masked but not interpolated. |
| Search Signal | $v_s$ | A working copy of $v_m$ used only for peak detection. NaN gaps are bridged via PCHIP interpolation. May optionally receive a human-gated secondary Butterworth filter. |

The key invariant: **peak locations are found on $v_s$; peak magnitudes are always read back from $v_m$.**

### 4.2 Optional Secondary Butterworth Filter

The Section 3 Diagnostics Panel displays the Welch PSD of each segment's velocity signal. If the PSD reveals high-frequency tracking jitter that impedes peak localization, the reviewer may activate a secondary low-pass Butterworth filter on $v_s$ via the interactive Section 4 Filter Gate. This decision is **human-gated** — the filter is never applied automatically. $v_m$ is unaffected regardless.

### 4.3 Peak Counting — Four Validation Gates

`scipy.signal.find_peaks` is run on $v_s$. A candidate peak is accepted as a valid movement pulse ($N_p$) only if it passes all four gates simultaneously:

1. **Noise Floor ($V$):** The peak magnitude (from $v_m$) must exceed the session-adaptive noise floor $V$, computed from a still reference window in the first 8 s of the recording (with a 20.0 mm/s static guard floor). Peaks at or below $V$ are indistinguishable from standing-still sensor noise.

2. **Prominence:** The peak must be locally prominent by at least $0.5 \times \sigma_{v_m}$ (half a standard deviation of the active signal). This excludes shallow ripples and sensor jitter from the count.

3. **Temporal Gate:** Adjacent peaks must be separated by at least **100 ms (12 frames at 120 Hz)**. This prevents a single broad velocity pulse from being double-counted due to fine surface irregularities.

4. **Artifact Exclusion:** Any peak whose frame is flagged as a MoCap tracking artifact (`{segment}__is_artifact = True`) is discarded. Artifact flags are set upstream in Step 06 when marker dropout, rotation rates, or velocities exceed validated thresholds.

### 4.4 Peaks Per Minute (PPM)

PPM normalizes the raw peak count by the time the segment was actually in motion:

$$\text{PPM} = \frac{N_p}{T_{active}} \times 60$$

where $T_{active}$ is the cumulative duration (in seconds) of frames where $v_m > V$ and the frame is not an artifact. This ensures that long pauses or artifact gaps do not artificially inflate or deflate the rate.

### 4.5 Smoothness — SPARC

The Spectral Arc Length (SPARC) measures movement smoothness as the arc length of the normalized velocity magnitude spectrum. A longer, more jagged arc (more negative SPARC) indicates less smooth, more fragmented movement.

SPARC is computed on $v_s$ (PCHIP-bridged to provide a continuous signal for FFT), with the integration range capped at the SG effective cutoff (~6.11 Hz) — the physical bandwidth limit of the velocity signal.

---

## 5. How to Work with the Notebook

**Setup (one time):**
```bash
pip install -r requirements.txt
```

**Running:**
1. Open `07_pulsicity_flow_4Review.ipynb` in JupyterLab.
2. **Run all cells in order** (Sections 1–7 have sequential dependencies).
3. **Section 2** lets you adjust peak detection parameters interactively before running the full pipeline.
4. **Section 3 + 4:** Review the PSD diagnostic plots for each segment. Activate the optional Butterworth filter per segment if you observe jitter above the SG cutoff that is not biomechanically meaningful. Click **"Confirm & Lock Filter Decisions"** when done.
5. **Section 5** provides an interactive windowed viewer — use the time slider and zoom controls to visually audit the detected peaks against the raw velocity trace before committing to batch export.
6. **Section 6** runs the full batch and writes one `*__pulsicity_metrics.parquet` per recording to the `output/` folder.
7. **Section 7** (optional) lets you log manual peak corrections for the scientific record.

---

## 6. Scientific Disclaimer — SPARC

SPARC is included as a **diagnostic/exploratory metric** in this pipeline. While it is computed rigorously — using PCHIP gap-bridging to avoid spectral leakage, and the SG −3 dB cutoff (~6.11 Hz) as the integration cap rather than an arbitrary 20 Hz ceiling — its interpretation for **unconstrained, whole-body Gaga movement** is exploratory.

SPARC was originally validated for goal-directed reaching tasks and constrained upper-limb movements. Its validity as a smoothness index for expressive, improvisational, full-body movement has not yet been established in the peer-reviewed literature. Results should be interpreted with appropriate caution and treated as hypothesis-generating rather than confirmatory.

---

## 7. File Index

| File | Description |
|---|---|
| `07_pulsicity_flow_4Review.ipynb` | Main analysis notebook |
| `src/pulsicity.py` | Backend computation module (Step 07) |
| `src/utils.py` | Standalone configuration (`DEMO_CFG`) |
| `src/__init__.py` | Package marker |
| `KINEMATIC_FEATURES_README.md` | Full Step 06 math reference (coordinate frames, Euler conventions, SG derivation) |
| `requirements.txt` | Python dependencies |
| `LICENSE` | Access terms |
| `data/` | Place the three Parquet files here (download separately) |
| `output/` | Step 07 output written here at runtime |
