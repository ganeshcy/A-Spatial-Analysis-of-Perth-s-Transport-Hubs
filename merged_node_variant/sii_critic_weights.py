"""
CRITIC weighting, and the collapsed frequency term
Perth Public Transport Network Analysis — merged_node_variant

CRITIC (Diakoulaki, Mavrotas & Papayannakis 1995) derives weights from the data
rather than asserting them. A criterion earns weight for two reasons:

    C_j = sigma_j * SUM_k (1 - r_jk)          w_j = C_j / SUM C

  * sigma_j — how much it VARIES. A criterion that is nearly the same everywhere
    cannot discriminate between stops and should not carry weight.
  * SUM (1 - r_jk) — how much it CONFLICTS with the others. A criterion that
    merely restates its neighbours adds nothing, so redundancy is penalised.

This is the data-driven arm of the Wu et al. [21] contrast, and it needs no
target variable — which matters here, because the supervised alternative (fitting
against network-efficiency loss) returned a cross-validated R2 of 0.114.

WHY THE FREQUENCY BLOCK FIRST
    F_AM, F_PM, F_off and F_rem correlate 0.88-0.99: they are close to one
    variable measured in four windows. That near-collinearity is exactly why
    collapsing them is low-risk — when four things are almost the same, how you
    combine them barely matters — and it makes the block a safe place to
    establish a method before extending it.

    The collapse also IMMUNISES the index against frequency inflation, which is
    why F_rem could be added at all. Because the block is reduced to a single
    term before the outer CRITIC sees it, adding frequency windows cannot add
    frequency weight: measured across one, three, four and five-term blocks the
    collapsed term's outer weight moves only between 0.1350 and 0.1361. The
    block's internal weights redistribute freely; its total does not grow.

Usage:
    python sii_critic_weights.py
    python sii_critic_weights.py --full     # also apply CRITIC to every component
"""

import os
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

# Weights currently in force, for comparison.
CURRENT = {"F_AM": .10, "F_PM": .10, "F_off": .05, "R": .20, "T": .20, "C": .35}
LOG_TRANSFORM = True     # must match sii_bus_stops_pipeline_merged.py


def zc(s, clip=3.0, mode="log"):
    """Standardise one criterion.

    mode="log"    log1p, then z, then clip  — matches the stored *_z columns
    mode="plain"  z, then clip              — no shape change
    mode="raw"    returned untouched
    """
    x = pd.to_numeric(s, errors="coerce").astype(float).fillna(0.0)
    if mode == "raw":
        return x
    if mode == "log" and LOG_TRANSFORM and x.min() >= 0:
        x = np.log1p(x)
    sd = x.std(ddof=0)
    if sd <= 1e-9 * max(1.0, abs(float(x.mean()))):
        return x * 0.0
    z = (x - x.mean()) / sd
    return z.clip(-clip, clip) if clip else z


def critic(df):
    """CRITIC weights for the columns of df. Returns (weights, sigma, conflict)."""
    # Min-max to [0,1] so sigma is comparable across criteria of different scale.
    X = df.apply(lambda c: (c - c.min()) / (c.max() - c.min())
                 if c.max() > c.min() else c * 0.0)
    sigma = X.std(ddof=0)
    R = X.corr(method="pearson").fillna(0.0)
    conflict = (1.0 - R).sum(axis=0)      # includes the self term (1-1)=0
    C = sigma * conflict
    return (C / C.sum()), sigma, conflict


def report(name, w, sigma, conflict):
    print(f"\n  {name}")
    print("    {:<16}{:>10}{:>12}{:>10}".format("crit", "sigma", "conflict", "weight"))
    for k in w.index:
        print("    {:<16}{:>10.4f}{:>12.3f}{:>10.4f}".format(
            k, sigma[k], conflict[k], w[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--standardise", choices=["log", "plain", "raw"],
                    default="log",
                    help="frame the criteria enter CRITIC in. 'log' (default) "
                         "matches the stored *_z columns; 'plain' is z without "
                         "log1p; 'raw' is untransformed. All frames are printed "
                         "for comparison whichever is adopted.")
    ap.add_argument("--t-spatial", type=int, default=None, metavar="METRES",
                    help="replace T (transfers.txt) with T_spatial at this "
                         "walk-distance threshold, from "
                         "transfer_potential_spatial.csv")
    args = ap.parse_args()

    p = os.path.join(OUT, "sii_bus_full.csv")
    d = pd.read_csv(p, low_memory=False)
    print(f"loaded {len(d):,} nodes from {os.path.basename(p)}")

    FBLOCK = ["F_AM", "F_PM", "F_off", "F_rem"]
    comps = ["F_AM", "F_PM", "F_off", "R", "T", "C"]   # legacy SII set
    t_label = "T"
    if args.t_spatial:
        ts = pd.read_csv(os.path.join(OUT, "transfer_potential_spatial.csv"),
                         low_memory=False)
        col = f"T_spatial_{args.t_spatial}"
        if col not in ts.columns:
            raise SystemExit(f"{col} not in transfer_potential_spatial.csv")
        d = d.drop(columns=["T"]).merge(
            ts[["stopnumber", col]].rename(columns={col: "T"}),
            on="stopnumber", how="left")
        d["T"] = d["T"].fillna(0.0)
        t_label = f"T_spatial@{args.t_spatial}m"
        print(f"  T replaced by {t_label} "
              f"(non-zero at {(d['T'] > 0).mean() * 100:.1f}% of nodes)")
    for c in comps + ["F_rem", "SII", "sched_trips_weekday"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # ---------------------------------------------------------------- block
    print("\n" + "=" * 68)
    print("CRITIC WITHIN THE FREQUENCY BLOCK")
    print("=" * 68)
    F = d[FBLOCK]
    print("\n  correlation inside the block:")
    print(F.corr().round(3).to_string().replace("\n", "\n    "))
    wF, sF, cF = critic(F)
    report("CRITIC weights within the block", wF, sF, cF)
    print("\n    inherited weights, 3-term block (renormalised): "
          f"F_AM {.10/.25:.3f}, F_PM {.10/.25:.3f}, F_off {.05/.25:.3f}"
          "   — F_rem had no inherited weight; it was omitted entirely")

    # Collapsed frequency term, in the SAME frame the other criteria will use —
    # otherwise the mix this script exists to avoid is reintroduced here.
    zF = pd.DataFrame({c: zc(d[c], mode=args.standardise)
                       for c in FBLOCK})
    d["F_collated"] = sum(wF[c] * zF[c] for c in wF.index)
    # A second copy fixed at log, used only for the SII-reproduction check below,
    # which must match the pipeline regardless of --standardise.
    d["F_collated_log"] = sum(wF[c] * zc(d[c], mode="log") for c in wF.index)

    # ------------------------------------------------------- does it cost?
    print("\n" + "=" * 68)
    print("DOES COLLAPSING THE BLOCK INTO ONE TERM COST ANYTHING?")
    print("=" * 68)
    # This comparison is against the STORED SII, so it is pinned to the log
    # frame the pipeline used, whatever --standardise says.
    z = {c: zc(d[c], mode="log") for c in comps}
    sii_3 = sum(CURRENT[c] * z[c] for c in comps)
    # the block's 0.25 now sits on the single collated term
    _fl = d["F_collated_log"]
    sii_1 = (0.25 * ((_fl - _fl.mean()) / _fl.std(ddof=0)).clip(-3, 3)
             + CURRENT["R"] * z["R"] + CURRENT["T"] * z["T"] + CURRENT["C"] * z["C"])

    wt = d["sched_trips_weekday"]
    print("\n  {:<34}{:>10}{:>11}".format("index", "Pearson", "Spearman"))
    for lab, s in [("three separate F terms", sii_3), ("one collated F term", sii_1)]:
        print("  {:<34}{:>10.4f}{:>11.4f}".format(
            lab, s.corr(wt), s.corr(wt, method="spearman")))
    print("\n  agreement between the two indices: Spearman {:.5f}".format(
        sii_3.corr(sii_1, method="spearman")))
    shift = (sii_3.rank(ascending=False) - sii_1.rank(ascending=False)).abs()
    print("  mean |rank shift| {:.2f} places   >50 places: {:,}   >200: {:,}".format(
        shift.mean(), int(shift.gt(50).sum()), int(shift.gt(200).sum())))
    print("\n  reproduces the stored SII: {}".format(
        bool((sii_3 - d["SII"]).abs().max() < 1e-9)))

    d[["stopnumber", "F_collated"]].to_csv(
        os.path.join(OUT, "f_collated.csv"), index=False)

    # ------------------------------------------------------------ full set
    if args.full:
        print("\n" + "=" * 68)
        print("CRITIC ACROSS EVERY COMPONENT  (preview — not adopted)")
        print("=" * 68)
        # A enters the SUMMED part of the index and so belongs in CRITIC.
        # L does NOT: it is multiplicative in HRI = S_hat * (1 + beta*L), so it
        # is not one of the criteria being weighted against each other.
        cols = ["F_collated", "R", "T", "C"]
        ap_ = os.path.join(OUT, "sii_bus_full_with_catchment.csv")
        if os.path.exists(ap_):
            a = pd.read_csv(ap_, usecols=["stopnumber", "A"], low_memory=False)
            d = d.merge(a, on="stopnumber", how="left")
            d["A"] = pd.to_numeric(d["A"], errors="coerce").fillna(0.0)
            cols.append("A")
            print(f"  A joined from sii_bus_full_with_catchment.csv "
                  f"(median {d['A'].median():,.0f})")
        else:
            print("  A not found — run the catchment stage to include it")
        # ------------------------------------------------------------------
        # ALL CRITERIA MUST ENTER CRITIC IN THE SAME FRAME.
        #
        # What CRITIC responds to, and what it ignores:
        #
        #   IGNORED — centring and scaling. critic() min-maxes each criterion
        #   before taking sigma, and min-max is invariant to affine transforms.
        #   Plain z-scoring is affine. So CRITIC on RAW criteria and CRITIC on
        #   PLAIN Z-SCORED criteria give the same weights, to 7e-4. Standardising
        #   does nothing here.
        #
        #   RESPONDED TO — distribution SHAPE. Only a non-affine transform (the
        #   log) or truncation (the clip) moves sigma, because only those change
        #   where the bulk sits relative to the extremes. On a variable with skew
        #   10, min-max is set by one outlier and the bulk collapses into a sliver.
        #
        # The original fault was therefore a LOG/NO-LOG mix, not a
        # standardised/raw mix: zc() log-transformed F_collated while R, T, C and
        # A arrived raw, so frequency took 0.449 of the index purely because its
        # competitors had had their sigma crushed by their own skew.
        #
        # Measured at T @ 500 m, four frames:
        #
        #   frame                 F      R      T      C      A   Spearman/board
        #   raw                0.104  0.192  0.373  0.188  0.143       0.514
        #   plain z, unclipped 0.103  0.192  0.374  0.188  0.144       0.659
        #   plain z, clip +-3  0.135  0.186  0.295  0.155  0.229       0.702
        #   log1p z, clip +-3  0.123  0.211  0.347  0.144  0.175       0.713
        #
        # Rows 1 and 2 share weights but not validation, because the weights are
        # affine-invariant and the INDEX is not: summing raw components lets A
        # (median 196, max 8,861) dominate numerically whatever weight it holds.
        #
        # WHY log1p IS THE DEFAULT. Raw skews run 5.5 to 10.6 and one stop sits
        # 31 SD from the mean on plain z. A z-score is only meaningful on a
        # roughly symmetric distribution; at skew 10 the +-3 clip discards real
        # information rather than trimming an artefact. log1p takes the F terms
        # to skew ~0.0-0.13, and it validates best of the four. Caveat: it barely
        # helps C (8.87 -> 6.82) — betweenness has a tail no monotone transform
        # fixes, so C is weakly standardised under any choice.
        #
        # --standardise {log,plain,raw} switches frame. The comparison row is
        # always printed: it is the evidence for the methodology chapter.
        # ------------------------------------------------------------------
        # NB: columns keep their plain names ("T", not "T_spatial@500m") so that
        # critic_weights.csv is machine-readable downstream. The threshold is
        # recorded separately, in the t_threshold column and in the printout.
        def frame(mode, clip=3.0):
            wf, _, _ = critic(d[FBLOCK])
            fc = sum(wf[c] * zc(d[c], clip=clip, mode=mode)
                     for c in wf.index)
            return pd.DataFrame({c: (fc if c == "F_collated"
                                     else zc(d[c], clip=clip, mode=mode))
                                 for c in cols})

        FRAMES = [("raw", "raw", None),
                  ("plain z, unclipped", "plain", None),
                  ("plain z, clip +-3", "plain", 3.0),
                  ("log1p z, clip +-3", "log", 3.0)]
        print("\n  ALL FRAMES  (min-max is affine-invariant, so 'raw' and "
              "'plain z, unclipped' must agree)")
        print("  " + "{:<22}".format("frame")
              + "".join("{:>9}".format(c[:8]) for c in cols)
              + "{:>13}".format("Spear/board"))
        store = {}
        for label, mode, clip in FRAMES:
            Zf = frame(mode, clip)
            wf_, sf_, cf_ = critic(Zf)
            Sf = sum(wf_[k] * Zf[k] for k in cols)
            store[label] = (wf_, sf_, cf_, Zf, Sf)
            print("  " + "{:<22}".format(label)
                  + "".join("{:>9.4f}".format(wf_[c]) for c in cols)
                  + "{:>13.4f}".format(Sf.corr(wt, method="spearman")))

        adopted = {"log": "log1p z, clip +-3", "plain": "plain z, clip +-3",
                   "raw": "raw"}[args.standardise]
        wA, sA, cA, std, S = store[adopted]
        wR = store["raw"][0]
        print()
        report(f"ADOPTED frame: {adopted}   (--standardise {args.standardise})",
               wA, sA, cA)
        print("\n    for comparison, current: F-block 0.25, R 0.20, T 0.20, C 0.35")

        print("\n    skew, raw -> log1p: " + ", ".join(
            f"{c} {d[c].skew():.1f}->{np.log1p(d[c].clip(lower=0)).skew():.1f}"
            for c in ["R", "T", "C", "A"] if c in d))
        print("\n  CRITIC-weighted index vs boardings: Pearson {:.4f}  Spearman {:.4f}".format(
            S.corr(wt), S.corr(wt, method="spearman")))
        sii_now = sum(CURRENT[c] * z[c] for c in comps)
        print("  index on CURRENT weights vs boardings: "
              "Pearson {:.4f}  Spearman {:.4f}".format(
                  sii_now.corr(wt), sii_now.corr(wt, method="spearman")))
        print("  agreement between the two: Spearman {:.4f}".format(
            S.corr(sii_now, method="spearman")))

        wout = pd.DataFrame({
            "criterion": list(wA.index),
            "sigma": [float(sA[k]) for k in wA.index],
            "conflict": [float(cA[k]) for k in wA.index],
            "weight": [float(wA[k]) for k in wA.index],
            "weight_raw_criteria": [float(wR[k]) for k in wA.index],
            "t_threshold": t_label,
            "standardise": args.standardise,
        })
        wpath = os.path.join(OUT, "critic_weights.csv")
        wout.to_csv(wpath, index=False)
        # The frequency sub-weights, so sii_assemble.py can rebuild F_collated.
        pd.DataFrame({"criterion": list(wF.index),
                      "weight": [float(wF[k]) for k in wF.index]}).to_csv(
            os.path.join(OUT, "critic_weights_fblock.csv"), index=False)
        print(f"\n  wrote {os.path.basename(wpath)} and critic_weights_fblock.csv")
        print(f"  T threshold embedded in these weights: {t_label}")


if __name__ == "__main__":
    main()
