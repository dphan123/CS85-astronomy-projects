# Project 24: data for the clickable 3D star map (prototypes/gaia_starmap/index.html).
#
# What this script does:
#   1. Loads the Gaia DR3 stars within 100 pc that prototypes/24_gaia_explorer.py downloaded
#      (run that script first; it caches them in data/gaia_100pc.csv).
#   2. Looks up each star in SIMBAD, the astronomers' database of named objects, to get its
#      catalogue name (e.g. "HD 172167"), its common name if it has one ("Vega") and its object type.
#   3. Adds the naked-eye stars (V < 5) within 100 pc from SIMBAD. Stars this bright saturate
#      Gaia's camera, so most of them (Vega, Sirius, Altair...) are missing from our Gaia sample.
#      Their distances come from SIMBAD, mostly measured by the earlier Hipparcos satellite.
#   4. Writes prototypes/gaia_starmap/stars.json with a random 60,000 Gaia stars, every named
#      Gaia star, and the naked-eye stars.
#
# Data: ESA Gaia DR3 (Gaia Collaboration, Vallenari et al. 2023), https://gea.esac.esa.int/archive/
#       SIMBAD (Wenger et al. 2000), CDS, Strasbourg. In November 2026 the main CDS site was partly
#       offline after a cybersecurity incident, so we use the official mirror at the CfA (Harvard).
#
# Run from the repo folder:   python3 prototypes/24_gaia_starmap.py
import io
import json
import os

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAIA_CACHE = os.path.join(ROOT, "data", "gaia_100pc.csv")
SIMBAD_IDS = os.path.join(ROOT, "data", "simbad_gaia_ids_100pc.csv")
SIMBAD_NAMES = os.path.join(ROOT, "data", "simbad_names_100pc.csv")
SIMBAD_BRIGHT = os.path.join(ROOT, "data", "simbad_bright_100pc.csv")
OUT = os.path.join(ROOT, "prototypes", "gaia_starmap", "stars.json")

# SIMBAD's TAP service. The usual address is https://simbad.cds.unistra.fr/simbad/sim-tap/sync;
# switch back to it once CDS has recovered.
SIMBAD_TAP = "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync"


def simbad_query(adql, cache_file):
    """Run one SIMBAD query (ADQL is SQL for astronomy) and cache the result as a CSV in data/."""
    if not os.path.exists(cache_file):
        r = requests.post(SIMBAD_TAP, timeout=600, data={
            "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "MAXREC": 2_000_000, "QUERY": adql})
        r.raise_for_status()
        with open(cache_file, "w") as f:
            f.write(r.text)
    return pd.read_csv(cache_file)


# ---------------------------------------------------------------- 1. Gaia stars
if not os.path.exists(GAIA_CACHE):
    raise SystemExit("Run prototypes/24_gaia_explorer.py first to download the Gaia stars.")
stars = pd.read_csv(GAIA_CACHE)
dist = 1000 / stars["parallax"]                        # parsecs
l, b = np.radians(stars["l"]), np.radians(stars["b"])
stars["x"] = dist * np.cos(b) * np.cos(l)              # +x towards the Galactic centre
stars["y"] = dist * np.cos(b) * np.sin(l)              # +y the direction the Sun orbits
stars["z"] = dist * np.sin(b)                          # +z Galactic north
stars["gaia_id"] = "Gaia DR3 " + stars["source_id"].astype(str)

# ---------------------------------------------------------------- 2. SIMBAD names and types
# Every SIMBAD object closer than ~105 pc (parallax > 9.5 mas) that has a Gaia DR3 identifier.
ids = simbad_query("""
    SELECT i.id AS gaia_id, b.main_id, b.otype
    FROM ident AS i JOIN basic AS b ON i.oidref = b.oid
    WHERE b.plx_value > 9.5 AND i.id LIKE 'Gaia DR3 %'""", SIMBAD_IDS)
# Their common names, which SIMBAD stores as identifiers starting with "NAME ".
names = simbad_query("""
    SELECT b.main_id, i.id AS name
    FROM ident AS i JOIN basic AS b ON i.oidref = b.oid
    WHERE b.plx_value > 9.5 AND i.id LIKE 'NAME %'""", SIMBAD_NAMES)
# SIMBAD pads some identifiers with extra spaces ("HD   8745"), so tidy them before matching.
tidy = lambda col: col.fillna("").str.split().str.join(" ")
ids["main_id"] = tidy(ids["main_id"])
names["main_id"] = tidy(names["main_id"])
names["name"] = tidy(names["name"]).str.replace("NAME ", "", regex=False)
# Pick one common name per object: SIMBAD's main name if that is a common name
# ("NAME Proxima Centauri"), otherwise the shortest one ("Sirius" rather than "Dog Star").
names["is_main"] = names["main_id"] == "NAME " + names["name"]
names["length"] = names["name"].str.len()
names = names.sort_values(["is_main", "length"], ascending=[False, True]).drop_duplicates("main_id")
names = names[["main_id", "name"]]

stars = stars.merge(ids.drop_duplicates("gaia_id"), on="gaia_id", how="left")
stars = stars.merge(names, on="main_id", how="left")
stars["main_id"] = stars["main_id"].fillna("")
for col in ["otype", "name"]:
    stars[col] = stars[col].fillna("")
print(f"{len(stars):,} Gaia stars; {(stars.main_id != '').sum():,} found in SIMBAD; "
      f"{(stars.name != '').sum():,} have a common name")

# ---------------------------------------------------------------- 3. naked-eye stars from SIMBAD
bright = simbad_query("""
    SELECT b.main_id, b.ra, b.dec, b.plx_value, b.otype, v.flux AS vmag, bb.flux AS bmag
    FROM basic AS b
    JOIN flux AS v ON v.oidref = b.oid AND v.filter = 'V'
    LEFT JOIN flux AS bb ON bb.oidref = b.oid AND bb.filter = 'B'
    WHERE b.plx_value > 10 AND v.flux < 5""", SIMBAD_BRIGHT)
bright["main_id"] = tidy(bright["main_id"])
bright = bright[~bright["main_id"].isin(stars["main_id"])]          # already in our Gaia sample
# Drop a whole system ("* alf Cen") when its member stars ("* alf Cen A", "* alf Cen B") are listed too.
is_system = bright["main_id"].apply(lambda m: (bright["main_id"].str.startswith(m + " ")).any())
bright = bright[~is_system].copy()
coords = SkyCoord(ra=bright["ra"].values * u.deg, dec=bright["dec"].values * u.deg).galactic
bdist = 1000 / bright["plx_value"]
bl, bb = coords.l.radian, coords.b.radian
bright["x"], bright["y"], bright["z"] = bdist * np.cos(bb) * np.cos(bl), bdist * np.cos(bb) * np.sin(bl), bdist * np.sin(bb)
bright["bv"] = bright["bmag"] - bright["vmag"]                     # B - V colour
# Rough conversion to Gaia's colour, only used to pick the dot's colour on the map
# (it matches the Sun, 0.65 -> 0.83, and Vega, 0.0 -> 0.05).
bright["bp_rp"] = 1.2 * bright["bv"].fillna(0.6) + 0.05
bright["abs_g"] = bright["vmag"] + 5 * np.log10(bright["plx_value"] / 1000) + 5   # absolute V magnitude
bright = bright.merge(names, on="main_id", how="left")
bright["name"] = bright["name"].fillna("")
print(f"Added {len(bright):,} naked-eye stars from SIMBAD that are not in the Gaia sample")

# ---------------------------------------------------------------- 4. choose stars and save
# A random 60,000 keeps the web page fast; we add every star with a common name so famous
# stars are always there to click on.
chosen = pd.concat([stars.sample(60_000, random_state=3), stars[stars["name"] != ""]]).drop_duplicates("source_id")
chosen["gaia"] = chosen["source_id"].astype(str)
chosen["bv"] = np.nan
chosen["vmag"] = np.nan
bright["gaia"] = ""                                    # empty Gaia ID marks a SIMBAD/Hipparcos star
chosen = pd.concat([chosen, bright], ignore_index=True)
nan_to_none = lambda col, nd: [None if pd.isna(v) else round(float(v), nd) for v in chosen[col]]
payload = {
    "source": "ESA Gaia DR3 (Gaia Collaboration, Vallenari et al. 2023); SIMBAD (Wenger et al. 2000), CDS",
    "total_stars": int(len(stars)), "shown": int(len(chosen)), "bright_added": int(len(bright)),
    "x": chosen["x"].round(2).tolist(), "y": chosen["y"].round(2).tolist(), "z": chosen["z"].round(2).tolist(),
    "colour": chosen["bp_rp"].round(3).tolist(),     # BP - RP colour: small = blue, large = red
    "abs_g": chosen["abs_g"].round(2).tolist(),      # absolute magnitude: small = intrinsically bright
    "gaia": chosen["gaia"].tolist(),
    "simbad": chosen["main_id"].fillna("").tolist(), "type": chosen["otype"].fillna("").tolist(),
    "name": chosen["name"].tolist(),
    "vmag": nan_to_none("vmag", 2), "bv": nan_to_none("bv", 2),   # only for the SIMBAD stars
}
with open(OUT, "w") as f:
    json.dump(payload, f, separators=(",", ":"))
print(f"Wrote {len(chosen):,} stars to {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
