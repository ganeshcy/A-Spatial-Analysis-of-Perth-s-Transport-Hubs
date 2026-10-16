"""
Choosing beta — diagnostics for the reliance amplification
Perth Public Transport Network Analysis — merged_node_variant

    HRI_i = S_plus_i * ( 1 + beta * L*_i )

WHAT KIND OF PARAMETER BETA IS
    beta is NOT estimable. There is no observable quantity it can be fitted
    against: boardings cannot serve, because the whole premise of the reliance
    term is that dependence is invisible in patronage where service is poor, so
    a beta fitted to boardings would be driven to zero by construction. Claiming
    an optimised beta would be claiming a ground truth that does not exist.

    beta is a DECLARED POLICY PARAMETER: a statement of how much a catchment's
    lack of alternatives should count relative to a hub's standing in the
    network. The defence is therefore not "beta = x is correct" but:

        (i)   here is exactly what beta does to the result,
        (ii)  here is the range over which the conclusions do not change,
        (iii) we report at beta = x and publish the sensitivity band.

    That is a defensible position. An unexplained beta is not.

FOUR DIAGNOSTICS, EACH ANSWERING A DIFFERENT QUESTION

  1. RANK DISPLACEMENT — "is the term doing anything?"
     Mean and 95th-percentile |rank change| between the supply-only ranking and
     HRI, as beta rises. At beta = 0 it is zero by definition. It climbs, then
     flattens once L* has taken over the ordering and further beta only rescales
     what it already decided. Useful beta is before the flattening.

  2. VARIANCE SHARE — "how much of the index IS reliance?"
     Decompose HRI = base + amp, where base = S_plus and amp = beta*S_plus*L*:

         Var(HRI) = Var(base) + Var(amp) + 2*Cov(base, amp)
         share_amp = ( Var(amp) + Cov(base, amp) ) / Var(HRI)

     The two shares sum to 1. This converts beta into a sentence a supervisor
     can accept or reject without knowing what beta is: "the reliance term
     accounts for one fifth of the variation in the index." The script inverts
     it — given a target share, it returns the beta that delivers it.

  3. TOP-100 CHURN — "who does it actually surface?"
     How many stops enter the top 100 that supply alone would have missed. This
     is the concrete output: the names of stops the index finds and SNAMUTS-style
     supply measures do not.

  4. EXEMPLAR TRAJECTORIES — "is it behaving sensibly?"
     Rank against beta for a handful of named stops spanning the typology. A
     captive stop should climb steadily; an integrated CBD hub should drift down
     slowly; nothing should move discontinuously. Discontinuity would indicate
     the rescale or the L* anchor is unstable.

OUTPUT
    beta_diagnostic.png     four panels
    beta_sensitivity.csv    the underlying table
    beta_exemplars.csv      trajectories of the named stops

Usage:
    python sii_beta_diagnostic.py
    python sii_beta_diagnostic.py --target-share 0.20
    python sii_beta_diagnostic.py --max-beta 3.0 --steps 61
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

INK = "#1F3864"
ACCENT = "#0E7C86"
WARM = "#C2410C"
GREY = "#94A3B8"


def hri(S, L, b):
    return S * (1.0 + b * L)


def var_share(S, L, b):
    """Share of Var(HRI) carried by the amplification term. Shares sum to 1."""
    base = S
    amp = b * S * L
    v = np.var(base + amp, ddof=0)
    if v <= 0:
        return 0.0
    cov = np.cov(base, amp, ddof=0)[0, 1]
    return float((np.var(amp, ddof=0) + cov) / v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-beta", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=41)
    ap.add_argument("--target-share", type=float, default=None,
                    help="report the beta giving this variance share to reliance")
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()

    p = os.path.join(OUT, "node_master.csv")
    if not os.path.exists(p):
        sys.exit("Missing node_master.csv — run sii_assemble.py first.")
    d = pd.read_csv(p, low_memory=False)
    S = d["S_plus"].to_numpy(float)
    L = d["L_star"].fillna(0.0).to_numpy(float)
    name = d["stopname"].astype(str).to_numpy()
    sub = d["suburb"].astype(str).to_numpy()

    print("=" * 74)
    print("BETA DIAGNOSTIC")
    print("=" * 74)
    print(f"  {len(d):,} nodes   L*: median {np.median(L):.3f}  "
          f"mean {L.mean():.3f}  share>0.5 {(L > 0.5).mean() * 100:.1f}%")

    betas = np.linspace(0.0, args.max_beta, args.steps)
    rank0 = pd.Series(S).rank(ascending=False, method="min").to_numpy()
    top0 = set(np.argsort(-S)[:args.topn])

    rows = []
    for b in betas:
        h = hri(S, L, b)
        rk = pd.Series(h).rank(ascending=False, method="min").to_numpy()
        shift = np.abs(rank0 - rk)
        topb = set(np.argsort(-h)[:args.topn])
        rows.append({
            "beta": round(float(b), 4),
            "mean_abs_rank_shift": round(float(shift.mean()), 2),
            "p95_abs_rank_shift": round(float(np.percentile(shift, 95)), 1),
            "spearman_vs_supply": round(float(pd.Series(S).corr(
                pd.Series(h), method="spearman")), 5),
            "reliance_var_share": round(var_share(S, L, b), 4),
            f"entrants_top{args.topn}": len(topb - top0),
            f"top{args.topn}_retained": len(topb & top0),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT, "beta_sensitivity.csv"), index=False)

    show = tbl[tbl["beta"].isin([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])]
    print("\n" + show.to_string(index=False))

    # ------------------------------------------------ invert the variance share
    print("\n" + "=" * 74)
    print("BETA AS A DECLARED VARIANCE SHARE")
    print("=" * 74)
    fine = np.linspace(0.0, max(args.max_beta, 4.0), 2001)
    shares = np.array([var_share(S, L, b) for b in fine])
    targets = ([args.target_share] if args.target_share
               else [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    print("  {:>16}{:>12}".format("reliance share", "beta"))
    for t in targets:
        if t <= shares.max():
            print("  {:>15.0%}{:>12.3f}".format(t, float(np.interp(t, shares, fine))))
        else:
            print("  {:>15.0%}{:>12}".format(t, "unreachable"))
    print(f"\n  ceiling: share -> {shares.max():.3f} as beta -> "
          f"{fine.max():.0f}; L* is bounded at 1, so the amplification cannot")
    print("  exceed a factor of (1+beta) and the share saturates.")

    # --------------------------------------------------------- stability band
    print("\n" + "=" * 74)
    print("STABILITY OF THE TOP 100")
    print("=" * 74)
    ref_b = 0.5
    ref = set(np.argsort(-hri(S, L, ref_b))[:args.topn])
    band = [b for b in betas
            if len(set(np.argsort(-hri(S, L, b))[:args.topn]) & ref)
            >= 0.9 * args.topn]
    if band:
        print(f"  >=90% of the beta=0.5 top {args.topn} is retained for beta in "
              f"[{min(band):.2f}, {max(band):.2f}]")
        print("  Conclusions stated over that range are insensitive to the choice.")

    # ------------------------------------------------------------- exemplars
    # Pick stops that DEMONSTRATE something, not stops that are extreme. The
    # highest-L* captive stops are tiny and sit near rank 10,000 whatever beta
    # does, so they show a flat line and teach nothing. What the panel is for is
    # the three behaviours the term should produce:
    #   surfaced  — largest rank GAIN at the reference beta; the stops the
    #               reliance term exists to find
    #   anchor    — top of the index; should stay put
    #   displaced — largest rank LOSS; well-served, well-substituted stops that
    #               the term correctly pushes down to make room
    ref_rank0 = pd.Series(S).rank(ascending=False, method="min").to_numpy()
    ref_rank_b = pd.Series(hri(S, L, 0.5)).rank(ascending=False,
                                                method="min").to_numpy()
    gain = ref_rank0 - ref_rank_b
    # restrict to stops that end up somewhere visible, else the log axis hides them
    visible = ref_rank_b <= 2000
    picks = []
    idx_g = np.where(visible)[0]
    picks += [("surfaced", int(i)) for i in idx_g[np.argsort(-gain[idx_g])[:2]]]
    picks += [("anchor", int(i)) for i in np.argsort(ref_rank_b)[:2]]
    idx_d = np.where(ref_rank0 <= 300)[0]
    picks += [("displaced", int(i)) for i in idx_d[np.argsort(gain[idx_d])[:2]]]

    ex_rows = []
    for lab, i in picks:
        for b in betas:
            r = pd.Series(hri(S, L, b)).rank(ascending=False, method="min")
            ex_rows.append({"class": lab, "stopnumber": int(d["stopnumber"].iloc[i]),
                            "stopname": name[i], "suburb": sub[i],
                            "beta": round(float(b), 4), "rank": int(r.iloc[i])})
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(os.path.join(OUT, "beta_exemplars.csv"), index=False)

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.suptitle("Choosing $\\beta$ — what the reliance term does to the index",
                 fontsize=14, color=INK, fontweight="bold")

    a = ax[0, 0]
    a.plot(tbl["beta"], tbl["mean_abs_rank_shift"], color=INK, lw=2, label="mean")
    a.plot(tbl["beta"], tbl["p95_abs_rank_shift"], color=ACCENT, lw=2,
           ls="--", label="95th percentile")
    a.set_title("1. Rank displacement vs supply-only ranking", color=INK)
    a.set_xlabel("$\\beta$"); a.set_ylabel("|rank change| (places)")
    a.legend(frameon=False)

    a = ax[0, 1]
    a.plot(tbl["beta"], tbl["reliance_var_share"] * 100, color=WARM, lw=2)
    for t in (10, 20, 30):
        a.axhline(t, color=GREY, lw=.8, ls=":")
        a.text(tbl["beta"].max(), t, f" {t}%", color=GREY, va="center", fontsize=8)
    a.set_title("2. Share of index variance carried by reliance", color=INK)
    a.set_xlabel("$\\beta$"); a.set_ylabel("share of Var(HRI)  (%)")

    a = ax[1, 0]
    a.plot(tbl["beta"], tbl[f"entrants_top{args.topn}"], color=ACCENT, lw=2)
    a.set_title(f"3. Stops entering the top {args.topn} on reliance alone", color=INK)
    a.set_xlabel("$\\beta$"); a.set_ylabel("entrants")

    a = ax[1, 1]
    cmap = {"surfaced": WARM, "anchor": INK, "displaced": ACCENT}
    for (lab, sn), g in ex.groupby(["class", "stopname"], sort=False):
        a.plot(g["beta"], g["rank"], lw=1.6, color=cmap[lab],
               label=f"{sn[:26]} ({lab})")
    a.set_yscale("log"); a.invert_yaxis()
    a.set_title("4. Exemplar trajectories (lower = better rank)", color=INK)
    a.set_xlabel("$\\beta$"); a.set_ylabel("HRI rank (log)")
    a.legend(frameon=False, fontsize=7, loc="lower left")

    for r in ax:
        for a in r:
            a.spines[["top", "right"]].set_visible(False)
            a.grid(alpha=.25, lw=.6)
            for b0 in (0.5,):
                a.axvline(b0, color=GREY, lw=.9, ls="--", zorder=0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(OUT, "beta_diagnostic.png")
    fig.savefig(png, dpi=170)
    print(f"\n  wrote {png}")
    print("  wrote beta_sensitivity.csv, beta_exemplars.csv")
    print("\n  Read panel 2 first: pick the share you are willing to defend in")
    print("  words, take the beta beside it, then check panel 3 tells a story")
    print("  you believe and panel 4 shows nothing moving discontinuously.")


if __name__ == "__main__":
    main()
