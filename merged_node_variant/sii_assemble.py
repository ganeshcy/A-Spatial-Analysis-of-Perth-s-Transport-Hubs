"""
Master node table, S_hat and the Hub Reliance Index
Perth Public Transport Network Analysis — merged_node_variant

WHAT THIS IS FOR
    Every component of the index currently lives in a different file, written at
    a different time, joined ad hoc by whatever script needs it. That is how
    sii_bus_full_with_catchment.csv came to be carrying T at 200 m three weeks
    after the pipeline moved to 500 m. This script is the single place the join
    happens, it records the vintage of every input, and it refuses to run if a
    T-dependent column would be read from a file written before the pipeline.

    Output: node_master.csv — one row per merged node, every component, every
    z-score, S_hat, HRI and the typology class. Everything downstream (GeoJSON
    export, thesis tables, the web map) reads this and nothing else.

THE INDEX

    S_hat_i = SUM_j w_j * z(X_ij)        over {F_collated, R, T, C, A}

    with w from CRITIC (critic_weights.csv) and z() the log1p standardisation
    with +/-3 clipping used throughout.

WHY S_hat IS RESCALED BEFORE AMPLIFICATION — this is not cosmetic

    S_hat is a weighted sum of z-scores, so it is centred near zero and 60% of
    nodes are NEGATIVE. The intended form

        HRI_i = S_hat_i * (1 + beta * L*_i)

    multiplies by a factor greater than one. On a negative S_hat that pushes the
    node FURTHER DOWN, so measured dependence makes a stop rank worse — the
    exact inversion of what the term is for. Measured at beta = 0.5:

        Morley Dr Before Randerson Pl, DIANELLA   S_hat -0.669  L* 1.00  ->  -1.004
        Wanneroo Rd After Ashley Rd, TAPPING      S_hat -0.362  L* 1.00  ->  -0.543

    384 nodes sit in that trap (S_hat < 0 and L* > 0.5), and they are precisely
    the captive class the index exists to surface.

    S_hat is therefore mapped to [0,1] before amplification:

        S_plus_i = ( S_hat_i - min S_hat ) / ( max S_hat - min S_hat )
        HRI_i    = S_plus_i * ( 1 + beta * L*_i )

    Min-max is safe HERE, unlike on the raw components, because S_hat is a sum
    of clipped z-scores and so is already bounded on both sides (-1.38 to 3.00);
    no single outlier can define the scale. The transform is strictly monotone,
    so it changes no ranking on S_hat alone: Spearman(S_hat, S_plus) = 1.000000.
    It only fixes the SIGN of the amplification.

    HRI is consequently on a ratio scale, not a z-scale, and is comparable
    within Perth only — L* is already anchored to Perth's own p99.

TYPOLOGY
    A 2x2 on S_plus and L*, cut at their medians by default:

        high S_plus, low  L*   integrated   busy, and well substituted
        high S_plus, high L*   critical     busy, and nothing else serves it
        low  S_plus, high L*   captive      thin service, no alternative, no car
        low  S_plus, low  L*   redundant    thin service, alternatives exist

Usage:
    python sii_assemble.py                    # beta from --beta, default 0.5
    python sii_assemble.py --beta 0.75
    python sii_assemble.py --cut tercile      # typology on terciles, not medians
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

Z_CLIP = 3.0
LOG_TRANSFORM = False          # must match sii_bus_stops_pipeline_merged.py

# Which columns come from which file, and whether that file is allowed to be
# older than sii_bus_full.csv. "tolerate_stale" is a claim that the columns taken
# do not depend on any parameter changed since — it must carry a reason.
SOURCES = {
    "sii_bus_full.csv": dict(
        cols=["stopnumber", "F_AM", "F_PM", "F_off", "R", "T", "C", "G",
              "lon", "lat", "stopname", "stoptype", "suburb", "F_rem",
              "sched_trips_weekday", "sched_trips_weekend", "quadrant"],
        tolerate_stale=False, reason="reference vintage"),
    "sii_bus_full_with_catchment.csv": dict(
        cols=["stopnumber", "A", "A_pop", "A_job", "A_source",
              "catchment_area_network", "catchment_suspect"],
        tolerate_stale=True,
        reason="A depends on catchment geometry and the F supply weights, "
               "not on T; its SII/T/T_z columns are NOT read"),
    "alternative_access.csv": dict(
        cols=["stopnumber", "S", "n_alternatives", "L"],
        tolerate_stale=True,
        reason="L is a supply-share ratio, independent of T"),
    "congestion_exposure.csv": dict(
        cols=["stopnumber", "G", "G_max_inflation", "G_segments",
              "G_peak_trips", "G_worst_period"],
        tolerate_stale=True,
        reason="G is rebuilt from GTFS running times and is independent of T; "
               "it OVERRIDES the G column carried in sii_bus_full.csv, which "
               "came from roscript3 and was 97.8% centrality"),
    "car_dependence.csv": dict(
        cols=["stopnumber", "zero_veh", "dwellings", "V_raw", "V",
              "L_reliance", "L_star"],
        tolerate_stale=True,
        reason="V is census-derived; L_reliance = L * V, both T-independent"),
}

# V is measured on a WIDER catchment than A and WITHOUT competitive allocation.
#
#   A  400 m walk, competitive allocation. A is a sum and must not exceed
#      Perth's population; competition is what stops it double-counting.
#   V  800 m walk, plain dasymetric overlay. V is a ratio, so overlap cancels —
#      and a household within reach of two stops depends on both, so splitting
#      it between them understates its dependence on each.
#
# 800 m is the upper bound of Australian best practice: the Climate Council's
# Next Stop Suburbia (2024) sets its Shared Transport Service Standard at
# 15-minute service within 800 m of every home and analyses Perth on exactly
# that basis with pedestrian-network catchments, citing Burke and Brown (2007)
# for "within 400 to 800 metres of every home is considered best practice".
# The Melbourne walking-access study (2024) supplies the behavioural reading:
# 400 m approximates the ACCEPTABLE walking distance for bus (60th percentile
# 580 m) and 800 m the TOLERABLE one (90th percentile 1.17 km) — and captive
# riders, who are what V measures, walk the tolerable distance rather than the
# acceptable one.
#
# Effect on the evidence base, which is the decisive argument:
#
#                             400 m competitive    800 m simple
#   median dwellings per stop            50              523
#   stops under 25 dwellings          2,847              968
#   stops under 5 dwellings             888              611
#
# Under the old bound a quarter of the network had a V that was essentially the
# shrinkage prior. Under the new one that falls to 8%. Rankings are largely
# preserved: Spearman(V_400, V_800simple) = 0.873, and for L* 0.893.
#
# CAUTION: dwellings_wide is an ~11x inflated denominator (the average dwelling
# is within 800 m of about eleven stops). It is NOT a count of households served
# and must never be reported as one. It cancels in the ratio; that is its only
# job.
V_OVERRIDE = "v_catchment_800_simple_fb.csv"


def zc(s, clip=Z_CLIP):
    """log1p standardisation with symmetric clipping — as used in the pipeline."""
    x = pd.to_numeric(s, errors="coerce").astype(float).fillna(0.0)
    if LOG_TRANSFORM and x.min() >= 0:
        x = np.log1p(x)
    sd = x.std(ddof=0)
    if sd <= 1e-9 * max(1.0, abs(float(x.mean()))):
        return x * 0.0
    return ((x - x.mean()) / sd).clip(-clip, clip)


def load_sources():
    ref = os.path.join(OUT, "sii_bus_full.csv")
    if not os.path.exists(ref):
        sys.exit("Missing sii_bus_full.csv — run the pipeline first.")
    ref_m = os.path.getmtime(ref)

    print("=" * 74)
    print("INPUT VINTAGES")
    print("=" * 74)
    print("  {:<42}{:<18}{}".format("file", "written", "vs reference"))

    frames, fatal = {}, []
    for fn, spec in SOURCES.items():
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            sys.exit(f"Missing {fn}")
        m = os.path.getmtime(p)
        age = "reference" if fn == "sii_bus_full.csv" else (
            "OK (newer)" if m >= ref_m - 1 else "OLDER")
        print("  {:<42}{:<18}{}".format(
            fn, time.strftime("%Y-%m-%d %H:%M", time.localtime(m)), age))
        if age == "OLDER":
            if spec["tolerate_stale"]:
                print("      tolerated: " + spec["reason"])
            else:
                fatal.append(fn)
        have = pd.read_csv(p, nrows=0).columns
        use = [c for c in spec["cols"] if c in have]
        missing = [c for c in spec["cols"] if c not in have]
        if missing:
            print(f"      note: absent from this file, skipped: {missing}")
        frames[fn] = pd.read_csv(p, usecols=use, low_memory=False)

    if fatal:
        sys.exit(f"\nSTALE AND NOT TOLERABLE: {fatal}\nRe-run the pipeline.")

    d = frames["sii_bus_full.csv"]
    for fn in SOURCES:
        if fn == "sii_bus_full.csv":
            continue
        f = frames[fn]
        dup = [c for c in f.columns if c != "stopnumber" and c in d.columns]
        if fn == "congestion_exposure.csv" and "G" in dup:
            # The rebuilt G deliberately replaces the pipeline's. Drop the old
            # one from the accumulator instead of dropping the new one.
            d = d.drop(columns=["G"])
            dup = [c for c in dup if c != "G"]
            print("  G taken from congestion_exposure.csv, not sii_bus_full.csv")
        if dup:
            f = f.drop(columns=dup)
            print(f"  dropped duplicate columns from {fn}: {dup}")
        d = d.merge(f, on="stopnumber", how="left")
    print(f"\n  assembled {len(d):,} nodes x {len(d.columns)} columns")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5,
                    help="reliance amplification; see sii_beta_diagnostic.py")
    ap.add_argument("--cut", choices=["median", "tercile"], default="median")
    args = ap.parse_args()

    d = load_sources()

    # ------------------------------------------------------------- weights
    wp = os.path.join(OUT, "critic_weights.csv")
    fp = os.path.join(OUT, "critic_weights_fblock.csv")
    if not (os.path.exists(wp) and os.path.exists(fp)):
        sys.exit("Missing critic_weights.csv / critic_weights_fblock.csv.\n"
                 "Run:  python sii_critic_weights.py --full --t-spatial 500")
    wf = pd.read_csv(fp).set_index("criterion")["weight"]
    wc = pd.read_csv(wp)
    t_thr = wc["t_threshold"].iloc[0] if "t_threshold" in wc.columns else "unknown"
    std_frame = wc["standardise"].iloc[0] if "standardise" in wc.columns else "log"
    here = "log" if LOG_TRANSFORM else "plain"
    if std_frame != here:
        print(f"\n  WARNING: weights were derived in the '{std_frame}' frame but this")
        print(f"  script standardises in the '{here}' frame. The two must match, or")
        print("  the weights describe a different set of criteria to the ones being")
        print("  weighted. Re-run sii_critic_weights.py --standardise " + here + ",")
        print("  or flip LOG_TRANSFORM here.")
    else:
        print(f"\n  frames consistent: weights and components both '{here}'")
    w = wc.set_index("criterion")["weight"]

    print("\n" + "=" * 74)
    print(f"WEIGHTS  (CRITIC, {std_frame}-standardised criteria)")
    print("=" * 74)
    print(f"  T threshold embedded in these weights: {t_thr}")
    print("\n  frequency block:  " + "  ".join(f"{k} {v:.4f}" for k, v in wf.items()))
    print("  index:            " + "  ".join(f"{k} {v:.4f}" for k, v in w.items()))
    if abs(w.sum() - 1.0) > 1e-6:
        sys.exit(f"  weights do not sum to 1 ({w.sum():.6f})")

    # ------------------------------------------------- standardised components
    for c in wf.index:
        d[f"{c}_z"] = zc(d[c])
    d["F_collated"] = sum(wf[c] * d[f"{c}_z"] for c in wf.index)

    crit = list(w.index)
    for c in crit:
        if c == "F_collated":
            continue
        d[f"{c}_z"] = zc(d[c])

    Z = {c: (d["F_collated"] if c == "F_collated" else d[f"{c}_z"]) for c in crit}
    d["S_hat"] = sum(w[c] * Z[c] for c in crit)

    lo, hi = d["S_hat"].min(), d["S_hat"].max()
    d["S_plus"] = (d["S_hat"] - lo) / (hi - lo)
    mono = d["S_hat"].corr(d["S_plus"], method="spearman")

    print("\n" + "=" * 74)
    print("S_hat AND THE RESCALE")
    print("=" * 74)
    print("  S_hat   min {:.3f}  median {:.3f}  max {:.3f}   negative: {:.1f}% of nodes"
          .format(lo, d["S_hat"].median(), hi, (d["S_hat"] < 0).mean() * 100))
    print("  S_plus  min {:.3f}  median {:.3f}  max {:.3f}"
          .format(d["S_plus"].min(), d["S_plus"].median(), d["S_plus"].max()))
    print(f"  rank-preserving: Spearman(S_hat, S_plus) = {mono:.6f}")
    if mono < 0.999999:
        sys.exit("  rescale altered the ranking — abort")
    trap = int(((d["S_hat"] < 0) & (d["L_star"] > 0.5)).sum())
    print(f"  nodes the rescale rescues (S_hat<0 and L*>0.5): {trap:,}")

    # ------------------------------------------------------------------ HRI
    b = args.beta
    # ---------------------------------------------------- wider-catchment V
    vp = os.path.join(OUT, V_OVERRIDE)
    if os.path.exists(vp):
        vw = pd.read_csv(vp, usecols=["stopnumber", "V_wide", "V_wide_raw",
                                      "dwellings_wide", "zero_veh_wide",
                                      "L_reliance_wide", "L_star_wide"],
                         low_memory=False)
        d = d.merge(vw, on="stopnumber", how="left")
        # NB: do not name these `a, b` — `b` is beta further down.
        for old_col, new_col in [
                ("V", "V_wide"), ("V_raw", "V_wide_raw"),
                ("dwellings", "dwellings_wide"), ("zero_veh", "zero_veh_wide")]:
            d[old_col + "_400m"] = d[old_col]
            d[old_col] = (pd.to_numeric(d[new_col], errors="coerce")
                          .fillna(d[old_col]))
        # L_reliance and L_star are RECOMPUTED here rather than imported, because
        # they depend on L as well as V. Importing them would freeze whatever L
        # existed when the V catchment was last traced, and the V trace is
        # expensive enough that it will not be re-run every time L changes.
        d["L_reliance_400m"] = d["L_reliance"]
        d["L_star_400m"] = d["L_star"]
        d["L_reliance"] = (pd.to_numeric(d["L"], errors="coerce").fillna(0.0)
                           * d["V"])
        q99 = float(np.percentile(d["L_reliance"], 99))
        d["L_star"] = ((d["L_reliance"] / q99).clip(0.0, 1.0)
                       if q99 > 0 else 0.0)
        print(f"    L_reliance = L x V recomputed here; p99 anchor {q99:.4f}, "
              f"{int((d['L_star'] >= 0.9999).sum())} nodes saturated")
        print(f"\n  V from {V_OVERRIDE} (800 m, no competitive allocation)")
        print("    400 m values retained as *_400m for the sensitivity table")
        print("    median V {:.4f}  median L* {:.4f}  stops <25 dwellings {:,}".format(
            d["V"].median(), d["L_star"].median(), int((d["dwellings"] < 25).sum())))
    else:
        print(f"\n  {V_OVERRIDE} not found — V stays on the 400 m catchment")

    d["G"] = pd.to_numeric(d.get("G"), errors="coerce").fillna(0.0)
    d["HRI"] = d["S_plus"] * (1.0 + b * d["L_star"].fillna(0.0))

    print("\n" + "=" * 74)
    print(f"HUB RELIANCE INDEX   (beta = {b})")
    print("=" * 74)
    print("  HRI  min {:.4f}  median {:.4f}  p99 {:.4f}  max {:.4f}".format(
        d["HRI"].min(), d["HRI"].median(), d["HRI"].quantile(.99), d["HRI"].max()))

    wt = pd.to_numeric(d["sched_trips_weekday"], errors="coerce")
    print("\n  {:<26}{:>10}{:>11}".format("vs observed boardings", "Pearson", "Spearman"))
    for lab, s in [("S_plus (supply side only)", d["S_plus"]), ("HRI", d["HRI"])]:
        print("  {:<26}{:>10.4f}{:>11.4f}".format(
            lab, s.corr(wt), s.corr(wt, method="spearman")))
    print("\n  NOTE: this is agreement with SCHEDULED TRIPS, not patronage — no")
    print("  stop-level patronage data exists for Perth (AUDIT_CHECKLIST item 8).")
    print("  Frequency is a component of the index, so it is a consistency check,")
    print("  not a validation, and a higher number here is not a better index.")

    # ------------------------------------------------------- rank displacement
    d["rank_S"] = d["S_plus"].rank(ascending=False, method="min").astype(int)
    d["rank_HRI"] = d["HRI"].rank(ascending=False, method="min").astype(int)
    d["rank_gain"] = d["rank_S"] - d["rank_HRI"]

    print("\n" + "=" * 74)
    print("WHAT THE RELIANCE TERM CHANGES  (this is the contribution)")
    print("=" * 74)
    print("  mean |rank shift| {:.1f} places   Spearman(S_plus, HRI) {:.4f}".format(
        d["rank_gain"].abs().mean(), d["S_plus"].corr(d["HRI"], method="spearman")))
    ent = d[(d["rank_HRI"] <= 100) & (d["rank_S"] > 100)]
    print(f"  enter the top 100 on reliance alone: {len(ent)}")
    if len(ent):
        print("\n  " + "{:<34}{:<14}{:>8}{:>8}{:>9}{:>9}".format(
            "stop", "suburb", "rank_S", "rank_HRI", "L*", "S_plus"))
        for _, r in ent.nsmallest(12, "rank_HRI").iterrows():
            print("  " + "{:<34}{:<14}{:>8}{:>8}{:>9.3f}{:>9.3f}".format(
                str(r["stopname"])[:33], str(r["suburb"])[:13],
                r["rank_S"], r["rank_HRI"], r["L_star"], r["S_plus"]))

    # ------------------------------------------------------------- typology
    if args.cut == "median":
        cs, cl = d["S_plus"].median(), d["L_star"].median()
        hi_s, hi_l = d["S_plus"] > cs, d["L_star"] > cl
    else:
        cs, cl = d["S_plus"].quantile(2 / 3), d["L_star"].quantile(2 / 3)
        hi_s, hi_l = d["S_plus"] >= cs, d["L_star"] >= cl

    d["typology"] = np.select(
        [hi_s & ~hi_l, hi_s & hi_l, ~hi_s & hi_l],
        ["integrated", "critical", "captive"], default="redundant")

    print("\n" + "=" * 74)
    print(f"TYPOLOGY  (cut: {args.cut} — S_plus {cs:.3f}, L* {cl:.3f})")
    print("=" * 74)
    tb = (d.groupby("typology")
            .agg(nodes=("stopnumber", "size"),
                 median_S=("S_plus", "median"),
                 median_Lstar=("L_star", "median"),
                 median_A=("A", "median"),
                 median_sched_trips=("sched_trips_weekday", "median"))
            .reindex(["integrated", "critical", "captive", "redundant"]))
    tb["share_%"] = (tb["nodes"] / len(d) * 100).round(1)
    print(tb.round(3).to_string())

    for cls in ["critical", "captive"]:
        sub = d[d["typology"] == cls].nlargest(5, "HRI")
        print(f"\n  top 5 {cls}:")
        for _, r in sub.iterrows():
            print("    {:<36}{:<14} HRI {:.3f}  L* {:.3f}".format(
                str(r["stopname"])[:35], str(r["suburb"])[:13],
                r["HRI"], r["L_star"]))

    # ----------------------------------------------------------------- write
    d["beta"] = b
    d["t_threshold"] = t_thr
    d["weight_frame"] = std_frame
    p = os.path.join(OUT, "node_master.csv")
    d.to_csv(p, index=False)
    print(f"\n  wrote {p}  ({len(d):,} nodes, {len(d.columns)} columns)")

    top = d.nlargest(100, "HRI")[
        ["stopnumber", "stopname", "suburb", "typology", "HRI", "S_plus",
         "L_star", "V", "A", "R", "T", "C", "rank_S", "rank_HRI", "sched_trips_weekday"]]
    top.to_csv(os.path.join(OUT, "hri_top100.csv"), index=False)
    print("  wrote hri_top100.csv")


if __name__ == "__main__":
    main()
