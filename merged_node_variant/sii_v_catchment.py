"""
V on a wider walk catchment than A
Perth Public Transport Network Analysis — merged_node_variant

THE ARGUMENT

    A and V ask different questions and should not share a distance threshold.

        A  how many people and jobs are within WALKING DISTANCE of this stop?
        V  of the households that DEPEND on this stop, what share have no car?

    A is correctly bounded at 400 m: it measures convenient walkable demand, and
    the PTA bus-stop standard is the right yardstick for that. V is not. A
    household 1,000 m from its nearest stop is in no 400 m catchment, so its car
    ownership enters V for no stop at all — yet that is exactly the household
    whose car ownership matters most. With a car the network is irrelevant to
    it; without one it walks the kilometre or does not travel. Bounding V at
    400 m therefore excludes the most transit-disadvantaged households from the
    measure of transit disadvantage.

    This is not only an argument from principle. Captive riders show measurably
    higher distance tolerance than choice riders: the willingness to walk is
    itself a function of whether you have an alternative, so a single threshold
    applied to both populations is wrong in a known direction.

    Measured consequence of the 400 m bound: 24.2% of catchments are allocated
    fewer than 25 dwellings and 7.5% fewer than five, so for a quarter of the
    network the shrunk V is close to the metropolitan prior rather than a local
    observation. 12.0% of metropolitan mesh blocks have NO stop within 400 m.

WHAT THIS DOES

    Re-traces the walk network at V_CATCHMENT_M (default 800 m) and recomputes V
    over the same dasymetric mesh-block surface. Catchments are written to their
    own file; the 400 m polygons are NOT overwritten, because A depends on them
    and Chapter 4 reports them.

THREE ALLOCATION MODES, AND WHY --simple IS THE RIGHT ONE FOR V

    default            supply-weighted competitive allocation, as used for A
    --no-supply-weight competitive allocation with the supply term dropped
    --simple           plain dasymetric overlay, no competition at all

    Competitive allocation exists because A IS A SUM and must not exceed Perth's
    actual population; without it the metropolitan total inflated to 2.0x
    residents and 2.8x jobs. That justification does not transfer to V, which is
    a RATIO. Overlap inflates zero_veh and dwellings by the same factor, so it
    cancels — and splitting a household between two stops actively understates
    its dependence on each, when in fact it depends on both.

    Worse, the competitive divisor 1 / SUM_j a_ij is a per-mesh-block measure of
    how CONTESTED a block is, and in a ratio that does not cancel across blocks:
    it makes contested blocks count less toward V than uncontested ones, for a
    reason unconnected to car ownership. See apportion_simple() for the full
    argument and the NEERABUP case that exposed it.

WHY NOT NEAREST-STOP ASSIGNMENT EITHER
    Tested, and it fails for a different reason. See sii_v_nearest.py:
    partitioning the surface so every mesh block belongs to its single nearest
    stop leaves 2,875 stops (24.4%) with zero dwellings, including Perth
    Busport, Elizabeth Quay Bus Stn, Perth Underground and Curtin University Bus
    Stn — in a dense area some other stop is always closer, so a major
    interchange is never anyone's nearest stop and vanishes from V. A partition
    answers "who is closest to this stop", which is not "who can reach it".

COST
    One bounded Dijkstra per node at 800 m over the cached OSM walk graph.
    Roughly four times the explored area of the 400 m trace, so budget well over
    an hour, and note the graph needs several GB of RAM to load.

Usage:
    python sii_v_catchment.py                 # 800 m, then recompute V
    python sii_v_catchment.py --metres 1000
    python sii_v_catchment.py --reuse --simple --fallback   # ADOPTED
"""

import os
import sys
import argparse
import tempfile
import importlib.util
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
GEO = os.path.join(OUT, "geojson")
DATA = os.path.join(BASE, "data", "abs")

G34_SA1 = os.path.join(DATA, "2021Census_G34_WA_SA1.csv")
ZERO_COL = "Num_MVs_per_dweling_0_MVs"
TOT_COL = "Num_MVs_per_dweling_Tot"

V_CATCHMENT_M = 800.0
SHRINKAGE_K = 25.0

_spec = importlib.util.spec_from_file_location(
    "cp", os.path.join(BASE, "sii_catchment_population_merged.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def apportion_simple(catchments, zones, label, quiet=False):
    """
    Plain dasymetric overlay: each stop is credited the share of every mesh
    block that falls inside its catchment, with NO normalisation across
    competing catchments.

        contrib_ij = count_i * a_ij          a_ij = share of zone i in catchment j

    This DOES NOT CONSERVE. Where catchments overlap, the same dwelling is
    credited to every stop that reaches it, so the network-wide sum exceeds the
    true total. For A that would be fatal — A is a sum, and the inflation it
    produced (2.0x residents, 2.8x jobs, unevenly) is exactly what competitive
    allocation was introduced to fix.

    For V it is not a defect but the correct behaviour, for two reasons.

    FIRST, V IS A RATIO. Overlap inflates the numerator and the denominator of
    zero_veh / dwellings by the same factor, so it cancels. Nothing is gained by
    dividing the households up, and something is lost: a household within
    walking distance of two stops genuinely depends on both, and splitting it
    between them understates its dependence on each.

    SECOND, COMPETITIVE ALLOCATION INTRODUCES AN ARBITRARY REWEIGHTING. Its
    divisor 1 / SUM_j a_ij is a per-mesh-block constant measuring how CONTESTED
    that block is. In a ratio it does not cancel across blocks — it makes
    contested blocks count less toward V than uncontested ones, for a reason
    that has nothing to do with car ownership. That is the mechanism behind the
    Joondalup Dr / NEERABUP result, where zero-vehicle dwellings held at ~8.5
    while total dwellings fell 149.5 -> 39.0 and V tripled on arithmetic alone.

    The overlap factor is reported so the non-conservation is explicit rather
    than hidden.
    """
    import geopandas as gpd

    if not quiet:
        print(f"  Apportioning {label} (plain dasymetric overlay, no competition) ...")

    cat = catchments[["stopnumber", "geometry"]].reset_index(drop=True)
    zon = zones[["zone_count", "zone_area", "geometry"]].reset_index(drop=True)

    joined = gpd.sjoin(cat, zon, how="inner", predicate="intersects")
    if joined.empty:
        print(f"    WARNING: no overlap between catchments and {label} zones.")
        return pd.Series(dtype=float, name=label)

    left = gpd.GeoSeries(cat.geometry.loc[joined.index].values, crs=cp.CRS_METRIC)
    right = gpd.GeoSeries(zon.geometry.loc[joined["index_right"]].values,
                          crs=cp.CRS_METRIC)
    inter = left.intersection(right).area.to_numpy()
    za = joined["zone_area"].to_numpy()
    a_ij = np.divide(inter, za, out=np.zeros_like(inter), where=za > 0)

    contrib = joined["zone_count"].to_numpy() * a_ij
    out = (pd.DataFrame({"stopnumber": joined["stopnumber"].to_numpy(),
                         "v": contrib})
           .groupby("stopnumber")["v"].sum().rename(label))

    reached = zon.iloc[sorted(set(joined["index_right"]))]["zone_count"].sum()
    if not quiet and reached > 0:
        print(f"    overlap factor {out.sum() / reached:.2f}x "
              f"(sum {out.sum():,.0f} over {reached:,.0f} reached) "
              f"— expected, and cancels in the ratio")
    return out


def main():
    import geopandas as gpd

    ap = argparse.ArgumentParser()
    ap.add_argument("--metres", type=float, default=V_CATCHMENT_M)
    ap.add_argument("--k", type=float, default=SHRINKAGE_K)
    ap.add_argument("--anchor", type=float, default=99.0)
    ap.add_argument("--reuse", action="store_true",
                    help="load previously traced polygons instead of re-tracing")
    ap.add_argument("--fallback", action="store_true",
                    help="substitute a radial disc of --metres for stops whose "
                         "walk trace failed. Strongly recommended; see below.")
    ap.add_argument("--simple", action="store_true",
                    help="plain dasymetric overlay, no competitive allocation. "
                         "Correct for a ratio; see apportion_simple().")
    ap.add_argument("--no-supply-weight", action="store_true",
                    help="allocate by area overlap only, dropping the supply term. "
                         "See the note below on why this may be right for V.")
    ap.add_argument("--limit", type=int, default=None,
                    help="trace only the first N stops (for a timing check)")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # SHOULD V's ALLOCATION BE SUPPLY-WEIGHTED?
    #
    # apportion_competitive splits each mesh block between the catchments
    # reaching it in proportion to (area overlap x supply). For A that is right:
    # it is the 2SFCA logic, and a household with a well-served stop and a poorly
    # served one nearby is drawn to the better-served one.
    #
    # For V the same weighting is questionable. V asks what share of the
    # households AROUND a stop have no car — a property of the place, not of
    # which stop wins a service competition. Supply-weighting removes denominator
    # from thinly-served stops, which are exactly the ones the index exists to
    # surface.
    #
    # The effect is not small, and widening the catchment amplifies it, because
    # more overlap means the supply term does more work. Measured 400 m -> 800 m:
    #
    #     stop supply quartile   median change in allocated dwellings   % losing
    #     lowest                              -1.2                        55%
    #     highest                            +18.3                        24%
    #
    # Joondalup Dr Before Wanneroo Rd (NEERABUP) is the clearest case: its
    # zero-vehicle dwellings barely moved (8.57 -> 8.43) while its total fell
    # 149.5 -> 39.0, so V rose 0.056 -> 0.151 purely because the denominator
    # collapsed. That is allocation arithmetic, not a measurement of reliance.
    #
    # --no-supply-weight allocates by area overlap alone. Conservation still
    # holds; only the split between competing catchments changes.
    # ------------------------------------------------------------------
    m = args.metres
    gpkg = os.path.join(GEO, f"catchments_network_{int(m)}.gpkg")
    os.makedirs(GEO, exist_ok=True)

    print("=" * 74)
    print(f"V ON A {m:.0f} m WALK CATCHMENT")
    print("=" * 74)

    if args.reuse and os.path.exists(gpkg):
        cat = gpd.read_file(gpkg).to_crs(cp.CRS_METRIC)
        print(f"  reusing {os.path.basename(gpkg)}: {len(cat):,} catchments")

    else:
        # network_catchments() reads CATCHMENT_M from the module at call time,
        # so overriding it here changes the Dijkstra cutoff without touching the
        # 400 m configuration that A uses.
        cp.CATCHMENT_M = m
        print(f"  tracing at {m:.0f} m (CATCHMENT_M overridden from 400) ...")
        stops = cp.load_stops(limit=args.limit)
        cat, _sens = cp.network_catchments(stops)
        cat = gpd.GeoDataFrame(cat, geometry="geometry", crs=cp.CRS_METRIC)
        cat.to_file(gpkg, driver="GPKG")
        print(f"  wrote {os.path.basename(gpkg)}")

    # ------------------------------------------------------------------
    # RADIAL FALLBACK FOR FAILED TRACES
    #
    # network_catchments() flags a stop suspect when the walk graph reaches
    # under MIN_REACH_M of street from it, or when the stop had to snap more
    # than MAX_SNAP_M to find a graph node. Those polygons describe a defect in
    # the OSM pedestrian data, not the built environment around the stop, and
    # measuring V over them is measuring nothing: at 800 m the 492 flagged
    # stops have a median catchment of 119,160 m2 against 742,811 for the rest,
    # and 447 of them are allocated under 25 dwellings, which is few enough
    # that shrinkage pulls V to the metropolitan prior.
    #
    # Catchment demand has always substituted a radial disc for these stops.
    # V did not, because this script was written after that decision and never
    # inherited it. The disc used here is --metres, so the fallback matches the
    # bound V is supposed to be measured at rather than A's narrower one.
    #
    # The trade is the standing one for any radial fallback: a disc reaches
    # across freeways, rail reserves and rivers that a pedestrian cannot cross,
    # so it OVERSTATES the catchment. That is the lesser error here — the
    # traced polygon understates it so badly that V carries no local signal at
    # all for two thirds of these stops — but it is a substitution to declare,
    # not one to make silently. The count is printed on every run.
    # ------------------------------------------------------------------
    if args.fallback:
        if "catchment_suspect" not in cat.columns:
            sys.exit("--fallback needs a catchment_suspect column; re-trace "
                     "without --reuse, or drop the flag.")
        sus = cat["catchment_suspect"].fillna(False).astype(bool)
        n = int(sus.sum())
        if n:
            pts = cp.load_stops()[["stopnumber", "geometry"]].to_crs(cp.CRS_METRIC)
            disc = pts.set_index("stopnumber").geometry.buffer(m)
            repl = cat.loc[sus, "stopnumber"].map(disc)
            before = cat.loc[sus].to_crs(cp.CRS_METRIC).geometry.area.median()
            cat.loc[sus, "geometry"] = repl.values
            after = cat.loc[sus].geometry.area.median()
            print(f"  radial fallback applied to {n:,} suspect stops "
                  f"({n / len(cat) * 100:.1f}%)")
            print(f"    median catchment {before:,.0f} -> {after:,.0f} m2 "
                  f"(a full {m:.0f} m disc is {3.14159 * m * m:,.0f} m2)")
        else:
            print("  --fallback requested but no stop is flagged suspect")
    else:
        if "catchment_suspect" in cat.columns:
            n = int(cat["catchment_suspect"].fillna(False).astype(bool).sum())
            if n:
                print(f"  WARNING: {n:,} stops have a failed walk trace and NO "
                      f"fallback. Pass --fallback.")

    # Supply weights for the competitive allocation, as at 400 m.
    base = pd.read_csv(os.path.join(OUT, "sii_bus_full.csv"), low_memory=False)
    fcols = [c for c in ("F_AM", "F_PM", "F_off", "F_rem") if c in base.columns]
    for c in fcols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    supply = base[fcols].sum(axis=1)
    supply.index = base["stopnumber"]

    mb = cp.load_mesh_blocks()
    g34 = pd.read_csv(G34_SA1)
    g34["SA1_CODE_2021"] = g34["SA1_CODE_2021"].astype(str).str.strip()

    tmpdir = tempfile.mkdtemp()
    tmp = os.path.join(tmpdir, "g34.csv")
    res = {}
    for label, col in [("zero_veh", ZERO_COL), ("dwellings", TOT_COL)]:
        g34[["SA1_CODE_2021", col]].to_csv(tmp, index=False)
        zones = cp.load_zones(cp.ASGS_GPKG, cp.SA1_LAYER, tmp, "SA1_CODE_2021",
                              col, "SA1_CODE_2021", f"SA1/{label}")
        surf = cp.build_weighted_surface(mb, zones, "SA1_CODE_2021",
                                         cp.POP_MB_WEIGHTS, label)
        if args.simple:
            res[label] = apportion_simple(cat, surf, label)
        else:
            res[label] = cp.apportion_competitive(
                cat, surf, label, supply=None if args.no_supply_weight else supply)
    try:
        os.remove(tmp); os.rmdir(tmpdir)
    except OSError:
        pass

    v = pd.DataFrame(res).fillna(0.0)
    p = v["zero_veh"].sum() / max(v["dwellings"].sum(), 1.0)
    v["V_wide_raw"] = np.divide(v["zero_veh"], v["dwellings"],
                                out=np.zeros(len(v)),
                                where=v["dwellings"].to_numpy() > 0)
    v["V_wide"] = (v["zero_veh"] + args.k * p) / (v["dwellings"] + args.k)
    v = v.rename(columns={"zero_veh": "zero_veh_wide",
                          "dwellings": "dwellings_wide"})
    v.index.name = "stopnumber"

    d = pd.read_csv(os.path.join(OUT, "node_master.csv"), low_memory=False)
    out = d[["stopnumber", "stopname", "suburb", "V", "dwellings", "zero_veh",
             "L", "L_reliance", "L_star"]].merge(
        v.reset_index(), on="stopnumber", how="left")
    for c in ("dwellings_wide", "zero_veh_wide", "V_wide", "V_wide_raw"):
        out[c] = out[c].fillna(0.0)

    out["L_reliance_wide"] = out["L"] * out["V_wide"]
    q = np.percentile(out["L_reliance_wide"], args.anchor)
    out["L_star_wide"] = ((out["L_reliance_wide"] / q).clip(0, 1)
                          if q > 0 else 0.0)
    tag = (f"{int(m)}" + ("_simple" if args.simple
                          else ("_area" if args.no_supply_weight else ""))
           + ("_fb" if args.fallback else ""))
    out.to_csv(os.path.join(OUT, f"v_catchment_{tag}.csv"), index=False)

    print("\n" + "=" * 74)
    print(f"400 m  vs  {m:.0f} m")
    print("=" * 74)
    print("  {:<34}{:>12}{:>14}".format("", "400 m", f"{m:.0f} m"))
    for lab, a, b in [
            ("dwellings counted, total", out.dwellings.sum(), out.dwellings_wide.sum()),
            ("median dwellings per stop", out.dwellings.median(), out.dwellings_wide.median()),
            ("stops with < 5 dwellings", (out.dwellings < 5).sum(), (out.dwellings_wide < 5).sum()),
            ("stops with < 25 dwellings", (out.dwellings < 25).sum(), (out.dwellings_wide < 25).sum())]:
        print("  {:<34}{:>12,.0f}{:>14,.0f}".format(lab, a, b))
    for lab, a, b in [("median V", out.V.median(), out.V_wide.median()),
                      ("max V", out.V.max(), out.V_wide.max()),
                      ("median L*", out.L_star.median(), out.L_star_wide.median())]:
        print("  {:<34}{:>12.4f}{:>14.4f}".format(lab, a, b))
    print("\n  rank agreement V: Spearman {:.4f}".format(
        out.V.corr(out.V_wide, method="spearman")))
    print("  rank agreement L*: Spearman {:.4f}".format(
        out.L_star.corr(out.L_star_wide, method="spearman")))

    out["shift"] = out.V_wide - out.V
    print("\n  LARGEST INCREASES — dependence the 400 m bound was hiding")
    for _, r in out.nlargest(10, "shift").iterrows():
        print("    {:<34}{:<14}V {:.3f} -> {:.3f}   dwellings {:>6.0f} -> {:>7.0f}".format(
            str(r.stopname)[:33], str(r.suburb)[:13], r.V, r.V_wide,
            r.dwellings, r.dwellings_wide))
    print(f"\n  wrote v_catchment_{tag}.csv"
          + ("   [area-weighted, no supply term]" if args.no_supply_weight else ""))
    print("  NOT adopted automatically — inspect, then wire V_wide into "
          "sii_car_dependence.py or sii_assemble.py.")


if __name__ == "__main__":
    main()
