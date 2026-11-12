# Project 24 stretch goal: go farther out and look for the Radcliffe Wave.
#
# The Radcliffe Wave is a ~2,700 pc long chain of star-forming gas clouds near the Sun that rises
# and falls above and below the Milky Way's disc like a wave. It was discovered at the CfA in 2020
# (Alves et al. 2020, Nature 578, 237) using 3D dust maps and cloud distances (Zucker et al. 2020).
# Konietzka et al. (2024, Nature 628, 62) later showed that the wave is oscillating.
#
# What this script does:
#   1. Downloads Gaia DR3 young, hot, luminous stars within ~1,700 pc (cached in data/).
#   2. Loads the published Radcliffe Wave tables from the Harvard Dataverse (cached in data/radcliffe/):
#      cloud distances, young star clusters, and the best-fit wave model.
#   3. Tests the idea with our own Gaia data: do the young stars near the wave's track rise and fall with it?
#   4. Saves two PNG plots in plots/.
#
# Run from the repo folder:   python3 prototypes/24_radcliffe_wave.py
import os

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
WAVE_DIR = os.path.join(DATA, "radcliffe")
YOUNG_CACHE = os.path.join(DATA, "gaia_young_1700pc.csv")
PLOTS = os.path.join(ROOT, "plots")

# ---------------------------------------------------------------- 1. Gaia young stars out to ~1,700 pc
# parallax > 0.6 mas      ->  closer than ~1,700 pc
# absolute G < 0          ->  at least ~100 times the Sun's light output
# BP - RP < 0.5           ->  blue, i.e. hot. Hot luminous stars burn out within ~100 million years,
#                             so they mark places where stars formed recently.
# Dust makes stars behind clouds look redder and fainter, so some young stars are missed.
query = """
SELECT source_id, l, b, parallax, bp_rp, phot_g_mean_mag + 5*LOG10(parallax) - 10 AS abs_g
FROM gaiadr3.gaia_source
WHERE parallax > 0.6 AND parallax_over_error > 10
  AND phot_g_mean_mag + 5*LOG10(parallax) - 10 < 0 AND bp_rp < 0.5
"""
if os.path.exists(YOUNG_CACHE):
    young = pd.read_csv(YOUNG_CACHE)
else:
    Gaia.ROW_LIMIT = -1
    young = Gaia.launch_job_async(query).get_results().to_pandas()
    assert len(young) < 50_000, "hit the archive's 50,000-row limit; split the query"
    young.to_csv(YOUNG_CACHE, index=False)
print(f"{len(young):,} young luminous Gaia stars within ~1,700 pc")


def galactic_xyz(l_deg, b_deg, dist_pc):
    """Sun-centred Galactic x (to the Galactic centre), y (direction of orbit), z (north), in parsecs."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return dist_pc * np.cos(b) * np.cos(l), dist_pc * np.cos(b) * np.sin(l), dist_pc * np.sin(b)


young["x"], young["y"], young["z"] = galactic_xyz(young["l"], young["b"], 1000 / young["parallax"])

# ---------------------------------------------------------------- 2. published Radcliffe Wave data
# Harvard Dataverse file IDs (see https://sites.google.com/cfa.harvard.edu/radcliffewave/data)
FILES = {
    "clusters_Konietzka2024.tab": 7211931,          # young star clusters (doi:10.7910/DVN/XSCB9N)
    "best_fit_model_Konietzka2024.tab": 8954305,    # best-fit wave model (doi:10.7910/DVN/F98QHY)
    "CO_clouds_Konietzka2024.tab": 7211929,         # clouds along the wave (doi:10.7910/DVN/Q7F4PC)
    "Handbook_Distances_Zucker2020.dat": 4361456,   # distances to all local clouds (doi:10.7910/DVN/07L7YZ)
}
os.makedirs(WAVE_DIR, exist_ok=True)
for name, file_id in FILES.items():
    path = os.path.join(WAVE_DIR, name)
    if not os.path.exists(path):
        r = requests.get(f"https://dataverse.harvard.edu/api/access/datafile/{file_id}?format=original", timeout=120)
        r.raise_for_status()
        open(path, "wb").write(r.content)

clusters = pd.read_csv(os.path.join(WAVE_DIR, "clusters_Konietzka2024.tab"))
model = pd.read_csv(os.path.join(WAVE_DIR, "best_fit_model_Konietzka2024.tab"))
wave_clouds = pd.read_csv(os.path.join(WAVE_DIR, "CO_clouds_Konietzka2024.tab"))
all_clouds = pd.read_csv(os.path.join(WAVE_DIR, "Handbook_Distances_Zucker2020.dat"), sep=r"\s+")
on_wave = clusters["inlier"] == 1
print(f"{len(wave_clouds)} clouds on the wave, {on_wave.sum()} wave clusters (+{(~on_wave).sum()} others), "
      f"{len(all_clouds)} local cloud sightlines")

# ---------------------------------------------------------------- 3. distance along the wave
# Seen from above, the wave is a long, nearly straight chain. For any object we find the closest point
# on the model's track (in x-y only) and record how far along the track that is, and how far off it.
mx, my = model["x"].values, model["y"].values
model["s"] = np.concatenate([[0], np.cumsum(np.hypot(np.diff(mx), np.diff(my)))])   # pc along the track


def along_wave(x, y):
    """Distance along the wave's track (pc) and sideways distance from it (pc) for points x, y."""
    d2 = (np.asarray(x)[:, None] - mx[None, :]) ** 2 + (np.asarray(y)[:, None] - my[None, :]) ** 2
    nearest = d2.argmin(axis=1)
    return model["s"].values[nearest], np.sqrt(d2[np.arange(len(nearest)), nearest]), nearest


for df in (young, clusters, wave_clouds):
    df["s"], df["off"], df["nearest"] = along_wave(df["x"], df["y"])

# Only draw the model where clouds or clusters were actually used to fit it (beyond that it is unconstrained).
covered = pd.concat([wave_clouds.loc[wave_clouds["off"] < 150, "s"], clusters.loc[on_wave & (clusters["off"] < 150), "s"]])
shown_model = model[(model["s"] >= covered.min() - 50) & (model["s"] <= covered.max() + 50)]

# Our own test: take Gaia young stars within 75 pc (sideways) of the track and compare their typical
# height z with the wave model's height at the same place.
near = young[(young["off"] < 75) & (young["s"] >= shown_model["s"].min()) & (young["s"] <= shown_model["s"].max())].copy()
bins = np.arange(0, model["s"].max() + 150, 150)
near["bin"] = pd.cut(near["s"], bins)
binned = near.groupby("bin", observed=True).agg(s=("s", "median"), z=("z", "median"), n=("z", "size")).reset_index()
binned = binned[binned["n"] >= 15]
model_z_at = np.interp(binned["s"], model["s"], model["z"])
corr = np.corrcoef(binned["z"], model_z_at)[0, 1]
print(f"{len(near):,} young stars within 75 pc of the track; correlation of their median height with "
      f"the model's height over {len(binned)} bins: r = {corr:.2f}")

# ---------------------------------------------------------------- 4a. plot: top-down view
SUN, WAVE, STAR, CLOUD = "#e8a317", "#c2410c", "#9aa3ad", "#2563eb"
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(young["x"], young["y"], s=1, color=STAR, alpha=0.5, label=f"Gaia young luminous stars ({len(young):,})")
other = all_clouds[~all_clouds["name"].isin(wave_clouds["name"])]
ax.scatter(other["x"], other["y"], s=10, facecolor="none", edgecolor="#64748b", lw=0.6,
           label="Other local clouds (Zucker+2020)")
sc = ax.scatter(wave_clouds["x"], wave_clouds["y"], c=wave_clouds["z"], cmap="RdBu_r", vmin=-180, vmax=180,
                s=34, edgecolor="k", lw=0.4, label="Radcliffe Wave clouds (colour = height z)")
ax.scatter(clusters.loc[on_wave, "x"], clusters.loc[on_wave, "y"], marker="^", s=22, color="#111827",
           label="Young clusters on the wave (Konietzka+2024)")
ax.plot(shown_model["x"], shown_model["y"], color=WAVE, lw=2, label="Best-fit wave model (Konietzka+2024)")
ax.add_patch(plt.Circle((0, 0), 100, fill=False, ls="--", color=SUN, lw=1.2))
ax.annotate("our 100 pc neighbourhood", (70, -95), color="#92400e", fontsize=8)
ax.scatter([0], [0], s=60, marker="*", color=SUN, edgecolor="k", zorder=5, label="Sun")
for name in ["Orion", "CMa_OB1", "Cepheus", "CygnusX", "NorthAmerica", "Perseus", "Taurus", "Lacerta"]:
    row = all_clouds[all_clouds["name"] == name].head(1) if name not in set(wave_clouds["name"]) else wave_clouds[wave_clouds["name"] == name]
    if len(row):
        ax.annotate(name.replace("_", " "), (row["x"].iloc[0], row["y"].iloc[0]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8.5, weight="bold")
ax.set_xlim(-1700, 1700); ax.set_ylim(-1700, 1700); ax.set_aspect("equal")
ax.set_xlabel("x (pc)  →  towards the Galactic centre"); ax.set_ylabel("y (pc)  →  direction the Sun orbits")
ax.set_title("The Radcliffe Wave seen from above the Milky Way's disc")
fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label="Cloud height above (+) / below (−) the disc, z (pc)")
ax.legend(loc="lower left", fontsize=8, markerscale=1.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "radcliffe_wave_top_view.png"), dpi=150)

# ---------------------------------------------------------------- 4b. plot: side view (the wave itself)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.scatter(near["s"], near["z"], s=3, color=STAR, alpha=0.55, label=f"Gaia young stars within 75 pc of the track ({len(near):,})")
ax.plot(binned["s"], binned["z"], "o-", color="#475569", ms=4, lw=1.2, label="Median height of those stars (150 pc bins)")
near_clouds = wave_clouds[wave_clouds["off"] < 150]
ax.scatter(near_clouds["s"], near_clouds["z"], s=36, color=CLOUD, edgecolor="k", lw=0.4, label="Radcliffe Wave clouds", zorder=4)
wc = clusters[on_wave & (clusters["off"] < 150)]
ax.scatter(wc["s"], wc["z"], marker="^", s=24, color="#111827", label="Young clusters on the wave", zorder=4)
ax.plot(shown_model["s"], shown_model["z"], color=WAVE, lw=2.5, label="Best-fit wave model", zorder=5)
ax.axhline(0, color="k", lw=0.6)
sun_s = along_wave([0], [0])[0][0]
ax.axvline(sun_s, color=SUN, lw=1, ls="--"); ax.annotate("closest point to the Sun", (sun_s, 160), xytext=(5, 0),
                                                         textcoords="offset points", color="#92400e", fontsize=8.5)
for name in ["CMa_OB1", "Orion", "Cepheus", "CygnusX"]:
    row = wave_clouds[wave_clouds["name"] == name]
    if len(row):
        ax.annotate(name.replace("_", " "), (row["s"].iloc[0], row["z"].iloc[0]), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8.5, weight="bold")
ax.set_ylim(-260, 200)
ax.set_xlabel("Distance along the wave (pc), from Canis Major (left) to Cygnus (right)")
ax.set_ylabel("Height above the disc, z (pc)")
ax.set_title("The Radcliffe Wave from the side: gas clouds and young clusters follow the wave,\n"
             f"but Gaia's bright blue stars as a group do not (correlation r = {corr:.2f})")
ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "radcliffe_wave_side_view.png"), dpi=150)
