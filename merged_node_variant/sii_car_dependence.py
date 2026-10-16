"""
Car dependence (V) and the composite reliance term
Perth Public Transport Network Analysis — merged_node_variant

    V_i          = zero-vehicle dwelling share of stop i's catchment
    L_reliance_i = L_service_i  x  V_i

WHY MULTIPLICATIVE, NOT BLENDED
    L_service and V answer different questions:

      L_service — is there ANOTHER STOP?      supply-side substitutability
      V         — can these people DRIVE?     demand-side dependence

    Averaging them produces a number where a high value could mean either, and
    the two imply different interventions. Reliance is the CONJUNCTION: you
    depend on a stop when there is no alternative service AND no ability to
    substitute a car. A stop with no alternatives in a high car-ownership suburb
    is not relied upon; neither is a car-less catchment with three good stops
    nearby. Only the conjunction scores high, which is what the Karner [16]
    equity-weighting argument in the review is after, and it keeps both factors
    separately inspectable in the output.

RESOLUTION
    G34 is available at SA1 (61,845 zones nationally) as well as SA2, so V is
    built at SA1 — unlike employment, which the ABS publishes no finer than SA2.
    A 400 m catchment sits inside a handful of SA1s, so this is a genuine
    catchment statistic rather than a suburb-wide average.

METHOD
    Numerator (zero-vehicle dwellings) and denominator (dwellings with vehicle
    counts stated) are apportioned SEPARATELY through the same dasymetric surface
    and competitive allocation used for A, then divided. Allocating the ratio
    directly would be wrong: it would weight a 3-dwelling SA1 the same as a
    300-dwelling one. Dwellings not stating a vehicle count are excluded from
    both, rather than assumed to have none.

SMALL-DENOMINATOR SHRINKAGE
    A raw ratio is meaningless where the denominator is tiny. Welshpool is an
    industrial estate: the dasymetric surface correctly finds almost no
    residential land there, so a catchment is allocated a FRACTION of one
    dwelling — and 0.08 / 0.11 then returns a confident-looking V = 0.75 built on
    no evidence at all. Before shrinkage all ten highest-reliance stops were
    Welshpool, every one at exactly 0.75, because they draw on the same one or
    two SA1s where 3 of 4 dwellings happen to be car-free.

    V is therefore an empirical-Bayes estimate, shrunk toward the metropolitan
    share in proportion to how little evidence supports it:

        V_i = (zero_veh_i + k * p) / (dwellings_i + k)

    A catchment with 400 dwellings is essentially untouched; one with 0.1
    collapses to p, which is the honest answer for a place where nobody lives.

RESCALING FOR USE IN THE INDEX
    L_reliance is a product of two small quantities and lands on a compressed,
    heavily right-skewed range: median 0.0088, p99 0.0575, max 0.2132. Entered
    directly into HRI = S_hat * (1 + beta * L), even beta = 1 would lift the most
    reliant stop by 21% and the median stop by under one per cent, which makes
    beta an arbitrary large number rather than a meaningful one.

    It is therefore rescaled onto [0,1] against the 99th percentile:

        L*_i = min( L_reliance_i / q99 , 1 )

    The p99 anchor is chosen over min-max because the maximum is a single stop
    (Perth Busport at 0.213, 3.7x the p99): min-max would let one node define the
    scale for the whole network, leaving 99.9% of stops with almost no
    amplification. It is chosen over a percentile rank because rank discards
    magnitude — the gap between the first and second most reliant stop would
    become 1/n regardless of how large it really is. The p99 anchor keeps
    magnitude within the bulk, is robust to one outlier, and saturates the 1% of
    stops that are genuinely the most reliant.

    Note this makes L* RELATIVE TO PERTH. A network that was uniformly more
    car-dependent would not show a higher L*, because the anchor moves with the
    data. That is acceptable for ranking stops within one city and should be
    stated whenever the index is compared across cities.

Usage:
    python sii_car_dependence.py
    python sii_car_dependence.py --k 50        # stronger shrinkage
    python sii_car_dependence.py --sweep       # report V and the ranking at several k
    python sii_car_dependence.py --anchor 95   # rescale against p95 instead
"""

import os
import sys
import tempfile
import importlib.util
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
GEO = os.path.join(OUT, "geojson")
DATA = os.path.join(BASE, "data", "abs")

G34_SA1 = os.path.join(DATA, "2021Census_G34_WA_SA1.csv")
ZERO_COL = "Num_MVs_per_dweling_0_MVs"      # ABS's own spelling
TOT_COL = "Num_MVs_per_dweling_Tot"

_spec = importlib.util.spec_from_file_location(
    "cp", os.path.join(BASE, "sii_catchment_population_merged.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


# Pseudo-count, in dwellings, for the shrinkage prior. It is the amount of
# evidence the metropolitan rate is worth: at k dwellings the prior and the local
# data carry equal weight. The median catchment holds ~50 dwellings, so k = 25
# leaves a typical catchment mostly data-driven while pulling near-empty ones
# firmly to the mean. Swept, not asserted.
SHRINKAGE_K = 25.0


def main():
    import argparse
    import geopandas as gpd

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=SHRINKAGE_K)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--anchor", type=float, default=99.0, metavar="PCTL",
                    help="percentile of L_reliance mapped to 1.0 (default 99)")
    args = ap.parse_args()

    for p in (G34_SA1, os.path.join(OUT, "sii_bus_full_with_catchment.csv"),
              os.path.join(OUT, "alternative_access.csv"),
              os.path.join(GEO, "catchments_network.gpkg")):
        if not os.path.exists(p):
            print(f"Missing {p}"); sys.exit(1)

    print("=" * 70)
    print("CAR DEPENDENCE (V) AND THE COMPOSITE RELIANCE TERM")
    print("=" * 70)

    base = pd.read_csv(os.path.join(OUT, "sii_bus_full_with_catchment.csv"),
                       low_memory=False)
    for c in ("F_AM", "F_PM", "F_off"):
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    supply = (base["F_AM"] + base["F_PM"] + base["F_off"])
    supply.index = base["stopnumber"]

    print("\n  loading catchments ...")
    net = gpd.read_file(os.path.join(GEO, "catchments_network.gpkg")).to_crs(cp.CRS_METRIC)
    rad = gpd.read_file(os.path.join(GEO, "catchments_radial.gpkg")).to_crs(cp.CRS_METRIC)
    fb = set(base.loc[base["A_source"] == "radial_fallback", "stopnumber"])
    cat = pd.concat([net[~net["stopnumber"].isin(fb)],
                     rad[rad["stopnumber"].isin(fb)]], ignore_index=True)
    cat = gpd.GeoDataFrame(cat, geometry="geometry", crs=cp.CRS_METRIC)
    print(f"    {len(cat):,} catchments ({len(fb):,} radial fallback)")

    mb = cp.load_mesh_blocks()

    # Zero-vehicle dwellings and stated-total dwellings, as two SA1 count layers.
    g34 = pd.read_csv(G34_SA1)
    g34["SA1_CODE_2021"] = g34["SA1_CODE_2021"].astype(str).str.strip()
    # A scratch CSV in the system temp dir, not in output/: load_zones() takes a
    # path, and deleting a file written onto a mounted volume can be refused.
    _tmpdir = tempfile.mkdtemp()
    tmp = os.path.join(_tmpdir, "g34_slice.csv")
    results = {}
    for label, col in [("zero_veh", ZERO_COL), ("dwellings", TOT_COL)]:
        g34[["SA1_CODE_2021", col]].to_csv(tmp, index=False)
        zones = cp.load_zones(cp.ASGS_GPKG, cp.SA1_LAYER, tmp,
                              "SA1_CODE_2021", col, "SA1_CODE_2021", f"SA1/{label}")
        surf = cp.build_weighted_surface(mb, zones, "SA1_CODE_2021",
                                         cp.POP_MB_WEIGHTS, label)
        results[label] = cp.apportion_competitive(cat, surf, label, supply=supply)
    try:
        os.remove(tmp); os.rmdir(_tmpdir)
    except OSError:
        pass

    v = pd.DataFrame(results).fillna(0.0)
    # Metropolitan prior: the allocated share across every catchment, so the
    # prior is drawn from the same surface as the data it shrinks.
    p_metro = v["zero_veh"].sum() / max(v["dwellings"].sum(), 1.0)
    v["V_raw"] = np.divide(v["zero_veh"], v["dwellings"],
                           out=np.zeros(len(v)), where=v["dwellings"].to_numpy() > 0)
    v["V"] = (v["zero_veh"] + args.k * p_metro) / (v["dwellings"] + args.k)
    print(f"\n  shrinkage: k = {args.k:g} dwellings, prior p = {p_metro:.4f}")
    print(f"    V_raw  median {v['V_raw'].median():.4f}  max {v['V_raw'].max():.4f}"
          f"  stops > 0.30: {int((v['V_raw'] > 0.3).sum()):,}")
    print(f"    V      median {v['V'].median():.4f}  max {v['V'].max():.4f}"
          f"  stops > 0.30: {int((v['V'] > 0.3).sum()):,}")

    L = pd.read_csv(os.path.join(OUT, "alternative_access.csv"))
    out = L[["stopnumber", "stopname", "suburb", "L", "n_alternatives"]].copy()
    out = out.merge(v.reset_index().rename(columns={"index": "stopnumber"}),
                    on="stopnumber", how="left")
    if args.sweep:
        print("\n  {:>6}{:>11}{:>10}{:>14}".format("k", "median V", "max V", "stops V>0.30"))
        for kk in (0, 10, 25, 50, 100):
            vv = (out["zero_veh"] + kk * p_metro) / (out["dwellings"] + kk)
            print("  {:>6g}{:>11.4f}{:>10.4f}{:>14,}".format(
                kk, vv.median(), vv.max(), int((vv > 0.3).sum())))
    for c in ("zero_veh", "dwellings", "V", "V_raw"):
        out[c] = out[c].fillna(0.0)
    out["L_reliance"] = out["L"] * out["V"]
    # Rescale onto [0,1] so that beta in HRI = S_hat * (1 + beta * L_star) reads
    # as the maximum proportional uplift reliance can confer.
    q = np.percentile(out["L_reliance"], args.anchor)
    out["L_star"] = (out["L_reliance"] / q).clip(0.0, 1.0) if q > 0 else 0.0
    out.to_csv(os.path.join(OUT, "car_dependence.csv"), index=False)

    metro = g34[ZERO_COL].sum() / g34[TOT_COL].sum()
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Greater Perth zero-vehicle share (SA1 totals): {metro:.4f}")
    print(f"  catchment V : median {out['V'].median():.4f}  mean {out['V'].mean():.4f}"
          f"  max {out['V'].max():.4f}")
    print(f"  L_service   : median {out['L'].median():.4f}")
    print(f"  L_reliance  : median {out['L_reliance'].median():.5f}"
          f"  max {out['L_reliance'].max():.4f}")
    print(f"  L_star (p{args.anchor:g} anchor = {q:.4f}) : median "
          f"{out['L_star'].median():.4f}  saturated at 1.0: "
          f"{int((out['L_star'] >= 0.9999).sum()):,}")
    print(f"  corr(L, V) = {out['L'].corr(out['V']):.3f}   "
          f"-> {'independent' if abs(out['L'].corr(out['V'])) < 0.3 else 'related'}")
    print()
    print("  Highest reliance (no alternative service AND low car access):")
    cols = ["stopname", "suburb", "L", "V", "L_reliance"]
    print(out.nlargest(10, "L_reliance")[cols].round(3).to_string(index=False))
    print()
    print("  The conjunction at work — high L but LOW reliance (car-owning catchment):")
    hi = out[out["L"] > out["L"].quantile(0.9)]
    print(hi.nsmallest(5, "V")[cols].round(3).to_string(index=False))
    print("\n  Wrote car_dependence.csv")


if __name__ == "__main__":
    main()
