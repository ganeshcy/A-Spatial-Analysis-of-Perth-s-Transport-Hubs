"""
V under nearest-stop assignment — the unbounded sensitivity case
Perth Public Transport Network Analysis — merged_node_variant

THE PROBLEM THIS TESTS

    V is computed over a 400 m walk catchment, the same polygon used for A. That
    is right for A, which asks how many people and jobs are within walking
    distance. It is wrong for V, which asks what share of the households that
    DEPEND on a stop have no car — because dependence is not bounded at 400 m.

    A household 1,000 m from its nearest stop sits in no catchment at all, so
    its car ownership enters V for no stop. Yet that is precisely the household
    whose car ownership matters most: with a car the network is irrelevant to
    it, and without one it either walks the kilometre or does not travel. The
    400 m bound therefore excludes the most transit-disadvantaged households
    from the measure of disadvantage — a bias pointing away from what the index
    is looking for.

    Measured on the 400 m version: 24.2% of catchments are allocated fewer than
    25 dwellings and 7.5% fewer than five, so for a quarter of the network V is
    close to the shrinkage prior rather than a local observation.

WHAT THIS DOES INSTEAD

    Every mesh block in the metropolitan surface is assigned to its nearest stop
    and contributes its whole dasymetric dwelling count to that stop. No
    threshold, no overlap, no competition: the assignment is a partition, so
    each dwelling is counted exactly once network-wide and every dwelling is
    counted somewhere. An isolated stop correctly inherits a large, thinly
    populated dependence area; a CBD stop inherits a small dense one.

    Distance is straight-line between the mesh-block centroid and the stop.
    Network distance would be preferable in principle, but at the scale that
    matters here — deciding which of two stops several hundred metres apart is
    nearer — the ranking rarely differs, and an unbounded network-distance
    nearest-neighbour search over every mesh block is far more expensive than
    the question warrants. Stated as an approximation, not hidden.

WHAT IT IS AND IS NOT FOR
    This is the SENSITIVITY case, not the adopted measure. A partition assigns
    every household to exactly one stop, which overstates exclusivity: in
    practice a household between two stops has some access to both. The 800 m
    walk catchment, which still allows competition between overlapping
    catchments, is the primary. This file exists to show how much the bound
    matters, and it is reported alongside rather than instead.

Usage:
    python sii_v_nearest.py
    python sii_v_nearest.py --k 25 --max-dist 5000
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DATA = os.path.join(BASE, "data", "abs")

G34_SA1 = os.path.join(DATA, "2021Census_G34_WA_SA1.csv")
ZERO_COL = "Num_MVs_per_dweling_0_MVs"
TOT_COL = "Num_MVs_per_dweling_Tot"

_spec = importlib.util.spec_from_file_location(
    "cp", os.path.join(BASE, "sii_catchment_population_merged.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def main():
    import geopandas as gpd
    import tempfile
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=25.0,
                    help="empirical-Bayes shrinkage, in dwellings")
    ap.add_argument("--max-dist", type=float, default=None, metavar="M",
                    help="optional cap on assignment distance; default unbounded")
    args = ap.parse_args()

    print("=" * 74)
    print("V UNDER NEAREST-STOP ASSIGNMENT")
    print("=" * 74)

    d = pd.read_csv(os.path.join(OUT, "node_master.csv"), low_memory=False)
    d = d.dropna(subset=["lon", "lat"])
    stops = gpd.GeoDataFrame(
        d[["stopnumber"]].copy(),
        geometry=gpd.points_from_xy(d["lon"], d["lat"]),
        crs="EPSG:4326").to_crs(cp.CRS_METRIC)
    sxy = np.c_[stops.geometry.x.values, stops.geometry.y.values]
    print(f"  {len(stops):,} stops")

    mb = cp.load_mesh_blocks()
    g34 = pd.read_csv(G34_SA1)
    g34["SA1_CODE_2021"] = g34["SA1_CODE_2021"].astype(str).str.strip()

    tmpdir = tempfile.mkdtemp()
    tmp = os.path.join(tmpdir, "g34.csv")
    surfaces = {}
    for label, col in [("zero_veh", ZERO_COL), ("dwellings", TOT_COL)]:
        g34[["SA1_CODE_2021", col]].to_csv(tmp, index=False)
        zones = cp.load_zones(cp.ASGS_GPKG, cp.SA1_LAYER, tmp,
                              "SA1_CODE_2021", col, "SA1_CODE_2021",
                              f"SA1/{label}")
        surfaces[label] = cp.build_weighted_surface(
            mb, zones, "SA1_CODE_2021", cp.POP_MB_WEIGHTS, label)
    try:
        os.remove(tmp); os.rmdir(tmpdir)
    except OSError:
        pass

    # The two surfaces do NOT share a row index: build_weighted_surface inner-joins
    # against its own zone set, and the SA1s reporting a zero-vehicle count are not
    # the same set as those reporting a total (15,630 vs 20,928 mesh blocks here).
    # Each surface is therefore assigned independently and aggregated by stop; the
    # two aggregates are joined on stopnumber, never positionally.
    tree = cKDTree(sxy)
    agg = {}
    for label in ("dwellings", "zero_veh"):
        s = surfaces[label].to_crs(cp.CRS_METRIC).reset_index(drop=True)
        cent = s.geometry.representative_point()
        dist, idx = tree.query(np.c_[cent.x.values, cent.y.values], k=1)
        keep = np.ones(len(s), bool)
        if args.max_dist:
            keep = dist <= args.max_dist
        a = pd.DataFrame({
            "stopnumber": stops["stopnumber"].to_numpy()[idx],
            "dist_m": dist,
            "val": pd.to_numeric(s["zone_count"], errors="coerce").fillna(0.0).to_numpy(),
        })[keep]
        tot_s, tot_a = float(s["zone_count"].sum()), float(a["val"].sum())
        print(f"  {label}: {len(s):,} mesh blocks, "
              f"conservation {tot_a:,.0f}/{tot_s:,.0f} "
              f"({tot_a / max(tot_s, 1) * 100:.2f}%)")
        if label == "dwellings":
            print(f"    assignment distance: median {np.median(dist):,.0f} m  "
                  f"p90 {np.percentile(dist, 90):,.0f} m  max {dist.max():,.0f} m")
            print(f"    mesh blocks whose nearest stop is beyond 400 m: "
                  f"{(dist > 400).mean() * 100:.1f}% — invisible to the 400 m V")
            if args.max_dist:
                print(f"    capped at {args.max_dist:,.0f} m: dropped "
                      f"{int((~keep).sum()):,}")
        agg[label] = a.groupby("stopnumber").agg(
            total=("val", "sum"), n_mb=("dist_m", "size"),
            med_m=("dist_m", "median"))

    v = (agg["dwellings"].rename(columns={"total": "dwellings_near",
                                          "n_mb": "mb_assigned",
                                          "med_m": "median_assign_m"})
         .join(agg["zero_veh"][["total"]].rename(columns={"total": "zero_veh_near"}),
               how="outer").fillna(0.0).reset_index())

    p = v["zero_veh_near"].sum() / max(v["dwellings_near"].sum(), 1.0)
    v["V_near_raw"] = np.divide(v["zero_veh_near"], v["dwellings_near"],
                                out=np.zeros(len(v)),
                                where=v["dwellings_near"].to_numpy() > 0)
    v["V_near"] = ((v["zero_veh_near"] + args.k * p)
                   / (v["dwellings_near"] + args.k))

    out = d[["stopnumber", "stopname", "suburb", "V", "dwellings",
             "L", "L_star"]].merge(v, on="stopnumber", how="left")
    for c in ("dwellings_near", "zero_veh_near", "V_near", "V_near_raw"):
        out[c] = out[c].fillna(0.0)
    out.to_csv(os.path.join(OUT, "v_nearest.csv"), index=False)

    print("\n" + "=" * 74)
    print("400 m CATCHMENT  vs  NEAREST-STOP ASSIGNMENT")
    print("=" * 74)
    print("  {:<30}{:>14}{:>16}".format("", "400 m catchment", "nearest-stop"))
    rows = [("dwellings counted, total", out["dwellings"].sum(),
             out["dwellings_near"].sum()),
            ("median dwellings per stop", out["dwellings"].median(),
             out["dwellings_near"].median()),
            ("stops with < 5 dwellings", (out["dwellings"] < 5).sum(),
             (out["dwellings_near"] < 5).sum()),
            ("stops with < 25 dwellings", (out["dwellings"] < 25).sum(),
             (out["dwellings_near"] < 25).sum())]
    for lab, a, b in rows:
        print("  {:<30}{:>14,.0f}{:>16,.0f}".format(lab, a, b))
    print("  {:<30}{:>14.4f}{:>16.4f}".format(
        "median V", out["V"].median(), out["V_near"].median()))
    print("  {:<30}{:>14.4f}{:>16.4f}".format(
        "max V", out["V"].max(), out["V_near"].max()))
    print("\n  rank agreement between the two: Spearman {:.4f}".format(
        out["V"].corr(out["V_near"], method="spearman")))

    out["shift"] = out["V_near"] - out["V"]
    print("\n  LARGEST INCREASES — stops whose dependence the 400 m bound hid")
    for _, r in out.nlargest(8, "shift").iterrows():
        print("    {:<34}{:<14}V {:.3f} -> {:.3f}   dwellings {:>6.0f} -> {:>7.0f}".format(
            str(r.stopname)[:33], str(r.suburb)[:13], r.V, r.V_near,
            r.dwellings, r.dwellings_near))
    print("\n  LARGEST DECREASES")
    for _, r in out.nsmallest(5, "shift").iterrows():
        print("    {:<34}{:<14}V {:.3f} -> {:.3f}   dwellings {:>6.0f} -> {:>7.0f}".format(
            str(r.stopname)[:33], str(r.suburb)[:13], r.V, r.V_near,
            r.dwellings, r.dwellings_near))
    print("\n  wrote v_nearest.csv")


if __name__ == "__main__":
    main()
