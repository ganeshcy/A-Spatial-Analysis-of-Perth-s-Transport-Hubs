"""
Export layers — QGIS GeoJSON and the web-map payload
Perth Public Transport Network Analysis — merged_node_variant

Replaces the baseline sii_geojson_export.py, which predates the merged
representation and carries only the original six components. Everything here is
read from node_master.csv and nothing else, so the exports cannot disagree with
the index.

TWO OUTPUTS, ONE SOURCE

  geojson/nodes_hri.geojson    full-fidelity, for QGIS. Every component, every
                               z-score, S_plus, HRI, typology, G. Style as
                               graduated points on HRI or any component.

  web/nodes.json               compact columnar payload for the web map. Same
                               nodes, rounded, stored as parallel arrays rather
                               than per-feature objects because a GeoJSON
                               FeatureCollection repeats every key name 11,773
                               times and quadruples the size for no benefit.

WHY THE WEB PAYLOAD CARRIES S_plus AND L_star RATHER THAN HRI

    HRI = S_plus * (1 + beta * L_star), and beta is an undecided policy
    parameter. Shipping a precomputed HRI would freeze one value of it. Shipping
    the two factors lets the page recompute the index, and the whole ranking, on
    every movement of a slider — which is the honest way to present a number
    whose sensitivity to beta is itself a finding.

Usage:
    python sii_export_merged.py
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
GEO = os.path.join(OUT, "geojson")
WEB = os.path.join(OUT, "web")

# Carried into QGIS at full precision.
GEO_FIELDS = ["stopnumber", "stopname", "suburb", "stoptype", "typology",
              "HRI", "S_plus", "S_hat", "L_star", "L_reliance", "L", "V",
              "F_AM", "F_PM", "F_off", "F_rem", "F_collated",
              "R", "T", "C", "A", "A_pop", "A_job", "G", "G_max_inflation",
              "n_alternatives", "sched_trips_weekday", "rank_S", "rank_HRI"]

# Carried into the browser. Kept deliberately short: every field costs ~11,773
# numbers. Anything derivable in JS (HRI, ranks) is NOT shipped.
WEB_NUM = {"s": ("S_plus", 4), "l": ("L_star", 4), "v": ("V", 3),
           "a": ("A", 0), "r": ("R", 0), "t": ("T", 0), "c": ("C", 4),
           "f": ("F_collated", 3), "g": ("G", 4),
           "am": ("F_AM", 0), "pm": ("F_PM", 0), "off": ("F_off", 0),
           "rem": ("F_rem", 0), "alt": ("n_alternatives", 0)}


def main():
    os.makedirs(GEO, exist_ok=True)
    os.makedirs(WEB, exist_ok=True)
    p = os.path.join(OUT, "node_master.csv")
    d = pd.read_csv(p, low_memory=False)
    d = d.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    print(f"loaded {len(d):,} nodes from node_master.csv")

    beta = float(d["beta"].iloc[0]) if "beta" in d.columns else 0.5
    frame = d["weight_frame"].iloc[0] if "weight_frame" in d.columns else "?"
    tthr = d["t_threshold"].iloc[0] if "t_threshold" in d.columns else "?"

    # ------------------------------------------------------------- GeoJSON
    # Thesis notation for the reader-facing layers. The pipeline keeps A for
    # catchment demand and L for alternative access internally; the exported
    # GeoJSON uses the symbols Chapter 3 defines, so a QGIS attribute table and
    # the thesis agree. Internal names are unchanged, so nothing upstream moves.
    # NB this is a SWAP: internal L (alternative access) exports as A, while
    # internal L_star (the rescaled reliance term) exports as L. It is safe
    # because a dict lookup is applied once per field, not chained — do not
    # reimplement it as sequential string replacement.
    RENAME = {"A": "D", "A_pop": "D_pop", "A_job": "D_job",
              "L": "A", "L_reliance": "L_reliance", "L_star": "L"}
    have = [c for c in GEO_FIELDS if c in d.columns]
    feats = []
    for r in d[have + ["lon", "lat"]].itertuples(index=False):
        rec = {RENAME.get(k, k): v for k, v in zip(have, r[:len(have)])}
        props = {}
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                props[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                props[k] = None if pd.isna(v) else round(float(v), 5)
            else:
                props[k] = None if pd.isna(v) else str(v)
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point",
                                   "coordinates": [round(float(r[-2]), 6),
                                                   round(float(r[-1]), 6)]},
                      "properties": props})
    gp = os.path.join(GEO, "nodes_hri.geojson")
    with open(gp, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    print(f"  wrote {os.path.relpath(gp, OUT)}  "
          f"{len(feats):,} features, {os.path.getsize(gp) / 1e6:.1f} MB, "
          f"{len(have)} fields")
    print("    thesis notation applied: " + ", ".join(
        f"{k}->{v}" for k, v in RENAME.items() if k in have))

    # --------------------------------------------------------- web payload
    def col(name, dp):
        x = pd.to_numeric(d[name], errors="coerce").fillna(0.0)
        return [int(v) if dp == 0 else round(float(v), dp) for v in x]

    payload = {
        "meta": {"n": len(d), "beta_default": beta, "weight_frame": frame,
                 "t_threshold": str(tthr),
                 "note": "HRI = s * (1 + beta * l), computed in the browser"},
        "name": d["stopname"].fillna("").astype(str).tolist(),
        "sub": d["suburb"].fillna("").astype(str).tolist(),
        "id": d["stopnumber"].astype(int).tolist(),
        "lon": [round(float(v), 5) for v in d["lon"]],
        "lat": [round(float(v), 5) for v in d["lat"]],
    }
    for short, (name, dp) in WEB_NUM.items():
        payload[short] = col(name, dp) if name in d.columns else [0] * len(d)

    wp = os.path.join(WEB, "nodes.json")
    with open(wp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    mb = os.path.getsize(wp) / 1e6
    print(f"  wrote {os.path.relpath(wp, OUT)}  {mb:.2f} MB "
          f"({len(WEB_NUM) + 5} arrays x {len(d):,})")
    if mb > 3:
        print("  WARNING: large for an inline payload; consider dropping fields")

    print(f"\n  beta at export {beta}   weight frame {frame}   T {tthr}")
    print("  typology: " + ", ".join(
        f"{k} {v:,}" for k, v in d["typology"].value_counts().items()))


if __name__ == "__main__":
    main()
