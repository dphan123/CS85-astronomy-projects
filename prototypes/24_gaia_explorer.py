# Project 24 Thursday prototype: an HR diagram linked to a 3D map of our stellar neighbourhood.
#
# What this script does:
#   1. Downloads every well-measured Gaia DR3 star within 100 parsecs (cached in data/).
#   2. Works out each star's 3D position around the Sun.
#   3. Saves two PNG plots in plots/ (HR diagram with white dwarfs highlighted, and their 3D map).
#   4. Writes prototypes/gaia_explorer/stars.json, the data for the interactive explorer page
#      (prototypes/gaia_explorer/index.html), where you lasso stars on the HR diagram.
#
# Data: ESA Gaia mission, Gaia DR3 (Gaia Collaboration, Vallenari et al. 2023), https://gea.esac.esa.int/archive/
#
# Run from the repo folder:   python3 prototypes/24_gaia_explorer.py
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
CACHE = os.path.join(ROOT, "data", "gaia_100pc.csv")
PLOTS = os.path.join(ROOT, "plots")
EXPLORER_DATA = os.path.join(ROOT, "prototypes", "gaia_explorer", "stars.json")

# ---------------------------------------------------------------- 1. download (only the first time)
query = """
SELECT source_id, bp_rp, phot_g_mean_mag + 5*LOG10(parallax) - 10 AS abs_g,
       parallax, l, b, ruwe
FROM gaiadr3.gaia_source
WHERE parallax > 10 AND parallax_over_error > 10
  AND phot_bp_mean_flux_over_error > 10 AND phot_rp_mean_flux_over_error > 10
  AND ruwe < 1.4
  AND random_index >= {lo} AND random_index < {hi}
"""
# parallax > 10 milliarcseconds  <=>  closer than 100 parsecs (~326 light-years).
# ruwe < 1.4 is Gaia's standard "the position measurement looks clean" filter. It removes most of
# the fake stars that piled up between the white dwarfs and the main sequence in the starter plot.

if os.path.exists(CACHE):
    stars = pd.read_csv(CACHE)
    print(f"Loaded {len(stars)} stars from {CACHE}")
else:
    # The archive returns at most 50,000 rows per query, so we download 8 random slices of the
    # sky catalogue (using Gaia's random_index column) and stack them together.
    Gaia.ROW_LIMIT = -1
    N_SLICES, MAX_INDEX = 8, 1_811_709_771
    edges = [round(i * MAX_INDEX / N_SLICES) for i in range(N_SLICES + 1)]
    pieces = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        piece = Gaia.launch_job_async(query.format(lo=lo, hi=hi)).get_results().to_pandas()
        assert len(piece) < 50_000, "a slice hit the archive's row limit; increase N_SLICES"
        pieces.append(piece)
        print(f"  slice {len(pieces)}/{N_SLICES}: {len(piece)} stars")
    stars = pd.concat(pieces, ignore_index=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    stars.to_csv(CACHE, index=False)
    print(f"Downloaded {len(stars)} stars and saved them to {CACHE}")

# ---------------------------------------------------------------- 2. 3D positions
# Distance in parsecs is 1000 / parallax (in milliarcseconds).
# l and b are Galactic longitude and latitude, so the Milky Way's disc is the x-y plane:
#   +x points to the Galactic centre, +y in the direction the Sun orbits, +z to the Galactic north pole.
dist = 1000 / stars["parallax"]
l, b = np.radians(stars["l"]), np.radians(stars["b"])
stars["x"] = dist * np.cos(b) * np.cos(l)
stars["y"] = dist * np.cos(b) * np.sin(l)
stars["z"] = dist * np.sin(b)

# ---------------------------------------------------------------- 3. pick out the white dwarfs
# White dwarfs are hot but tiny, so they sit far below the main sequence on the HR diagram.
# This straight line separates them from main-sequence stars of the same colour.
stars["white_dwarf"] = stars["abs_g"] > 3.33 * stars["bp_rp"] + 9.5
wd = stars[stars["white_dwarf"]]
print(f"{len(wd)} white dwarfs out of {len(stars)} stars ({100 * len(wd) / len(stars):.1f}%)")

GREY, ACCENT = "#9aa3ad", "#2a6fdb"

# Plot A: HR diagram with the white dwarfs highlighted
fig, ax = plt.subplots(figsize=(6.5, 7.5))
ax.hexbin(stars["bp_rp"], stars["abs_g"], gridsize=250, bins="log", cmap="Greys", mincnt=1)
ax.scatter(wd["bp_rp"], wd["abs_g"], s=1.5, color=ACCENT, label=f"White-dwarf candidates ({len(wd):,})")
line_x = np.array([-0.6, 1.6])
ax.plot(line_x, 3.33 * line_x + 9.5, ls="--", lw=1, color="#555", label="Dividing line used")
ax.invert_yaxis()
ax.set_xlabel("Colour BP − RP (blue  →  red)")
ax.set_ylabel("Absolute G magnitude (bright  →  up)")
ax.set_title(f"Gaia DR3: {len(stars):,} stars within 100 pc (RUWE < 1.4)")
legend = ax.legend(loc="upper right", frameon=False)
for handle in legend.legend_handles:              # bigger dot in the legend only
    if hasattr(handle, "set_sizes"): handle.set_sizes([30])
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "gaia_hr_white_dwarfs.png"), dpi=150)

# Plot B: where those white dwarfs are, in 3D around the Sun
background = stars.sample(20_000, random_state=1)    # a random sample keeps the plot readable
fig = plt.figure(figsize=(8, 7.5))
ax = fig.add_subplot(projection="3d")
ax.scatter(background["x"], background["y"], background["z"], s=0.6, color=GREY, alpha=0.35,
           label="All stars (random 20,000)")
ax.scatter(wd["x"], wd["y"], wd["z"], s=0.8, color=ACCENT, alpha=0.6, label=f"White-dwarf candidates ({len(wd):,})")
ax.scatter([0], [0], [0], s=80, color="#e8a317", edgecolor="k", label="Sun", depthshade=False)
ax.set_xlabel("x (pc) → Galactic centre"); ax.set_ylabel("y (pc) → direction of orbit"); ax.set_zlabel("z (pc) → north")
ax.set_box_aspect((1, 1, 1))
ax.set_title("Where the white dwarfs are: a sphere 200 pc across, centred on the Sun")
legend = ax.legend(loc="upper left", frameon=False)
for handle, size in zip(legend.legend_handles, [20, 20, 60]):   # same-sized legend dots
    handle.set_sizes([size]); handle.set_alpha(1)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "gaia_3d_white_dwarfs.png"), dpi=150)

# ---------------------------------------------------------------- 4. data for the interactive page
# A random 60,000 stars keeps the web page fast while keeping every region of the HR diagram.
sample = stars.sample(60_000, random_state=2)
columns = {
    "bp_rp": sample["bp_rp"].round(3), "abs_g": sample["abs_g"].round(2),
    "x": sample["x"].round(1), "y": sample["y"].round(1), "z": sample["z"].round(1),
}
payload = {
    "source": "ESA Gaia DR3 (Gaia Collaboration, Vallenari et al. 2023), gea.esac.esa.int/archive",
    "total_stars": int(len(stars)), "sample_size": int(len(sample)),
    **{name: values.tolist() for name, values in columns.items()},
}
with open(EXPLORER_DATA, "w") as f:
    json.dump(payload, f, separators=(",", ":"))
print(f"Wrote {len(sample)} stars to {EXPLORER_DATA} ({os.path.getsize(EXPLORER_DATA) / 1e6:.1f} MB)")
