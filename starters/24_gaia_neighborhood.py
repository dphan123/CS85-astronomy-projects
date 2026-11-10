# pip install astroquery matplotlib pandas
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

query = """
SELECT bp_rp, phot_g_mean_mag + 5*LOG10(parallax) - 10 AS abs_g
FROM gaiadr3.gaia_source
WHERE parallax > 10 AND parallax_over_error > 10
  AND phot_bp_mean_flux_over_error > 10 AND phot_rp_mean_flux_over_error > 10
  AND random_index >= {lo} AND random_index < {hi}
"""  # parallax > 10 mas  <=>  closer than 100 parsecs (~326 light-years)

# The archive returns at most 50,000 rows per query, but ~300,000 stars match.
# Every Gaia star has a random_index (0 to ~1.81 billion), so we split that range
# into 8 slices, download each slice (~38,000 stars), and stack them together.
Gaia.ROW_LIMIT = -1                        # don't let astroquery cut the results short either
N_SLICES, MAX_INDEX = 8, 1_811_709_771
edges = [round(i * MAX_INDEX / N_SLICES) for i in range(N_SLICES + 1)]
pieces = []
for lo, hi in zip(edges[:-1], edges[1:]):
    piece = Gaia.launch_job_async(query.format(lo=lo, hi=hi)).get_results().to_pandas()  # anonymous
    assert len(piece) < 50_000, "a slice hit the archive's row limit; increase N_SLICES"
    pieces.append(piece)
stars = pd.concat(pieces, ignore_index=True)
print(len(stars), "stars within 100 pc")

plt.figure(figsize=(6, 7))
plt.hexbin(stars["bp_rp"], stars["abs_g"], gridsize=250, bins="log", cmap="viridis", mincnt=1)
plt.gca().invert_yaxis()
plt.xlabel("Colour BP - RP (blue  ->  red)"); plt.ylabel("Absolute G magnitude (bright  ->  up)")
plt.title("Hertzsprung-Russell diagram: Gaia DR3 stars within 100 pc")
plt.savefig("gaia_neighborhood.png", dpi=150)
