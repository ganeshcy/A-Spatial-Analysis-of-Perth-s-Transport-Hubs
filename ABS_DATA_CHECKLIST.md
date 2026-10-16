# ABS data checklist — inputs for `sii_catchment_population.py`

Everything goes in `Research Proj/data/abs/`. The script checks for all of it up
front and lists whatever is missing before doing any work, so it is safe to run
early just to see where you stand.

```
Research Proj/
  data/
    abs/
      ASGS_2021_Main_Structure_GDA2020.gpkg   <- 1  required (SA1 + SA2 + MB)
      2021Census_G01_WA_SA1.csv               <- 2  required
      2021Census_W01B_WA_POW_SA2.csv          <- 3  required
      2021Census_W01A_WA_POW_SA2.csv          <-    not read (see item 3)
    cache/                                     <- created for the OSM graph
```

**Status: all present and verified working.** The paths and column names in the
script match these files. `--population-only` will run without item 3, but it
prints a warning and the result is provisional — see item 3 for why.

## Why not just use 2026 data?

You can download ASGS **Edition 4** boundaries — they began releasing 22 July
2026. But there is no 2026 Census data to pair them with. Census night is
11 August 2026; first data release is June 2027, employment-related data
October 2027, and journey-to-work data early 2028. So the counts you need are
the 2021 ones, and those are coded to Edition 3 geography. Downloading Ed4
boundaries and joining 2021 counts to them will silently drop or misallocate
every zone that changed between editions.

---

## 1. Boundaries — required

**Get:** `ASGS_2021_Main_Structure_GDA2020.gpkg` (~930 MB)

This is **one** GeoPackage containing every level as a separate layer — there is
no separate SA1 file and no separate SA2 file:

| Layer | Features |
|---|---|
| `MB_2021_AUST_GDA2020` | 368,286 |
| `SA1_2021_AUST_GDA2020` | 61,845 |
| `SA2_2021_AUST_GDA2020` | 2,473 |
| `SA3` / `SA4` / `GCCSA` / `STE` / `AUS` | — |

CRS is EPSG:7844 (GDA2020 geographic). Every layer carries `GCCSA_CODE_2021`, so
the script pushes a `GCCSA_CODE_2021 = '5GPER'` filter down into the driver and
never loads the other ~57,000 SA1s. Greater Perth resolves to **4,822 SA1s**
(2,116,468 residents) and **185 SA2s**.

From the ABS [digital boundary files page](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files),
under **Main Structure and Greater Capital City Statistical Areas**, take the
GeoPackage in **GDA2020**. Same files are mirrored on
[data.gov.au](https://data.gov.au/data/dataset/asgs-edition-3-2021-boundaries)
if the ABS download is slow.

> **Use Edition 3, not Edition 4.** Edition 3 is the vintage the 2021 Census was
> coded to. Edition 3's nominal currency period ends June 2026, so a newer
> edition may now be the default download — but pairing newer boundaries with
> 2021 counts will silently misalign zones where SA1s were split or merged. The
> filename must say `2021`.

If you ever swap this for shapefiles, set `ASGS_GPKG` to the `.shp` path and
blank out `SA1_LAYER` / `SA2_LAYER` — but the single GeoPackage is simpler and is
what the script is configured for.

## 2. SA1 population — required

**Get:** `2021Census_G01_WA_SA1.csv`

From [Census DataPacks](https://www.abs.gov.au/census/find-census-data/datapacks):

- Census year: **2021**
- DataPack type: **General Community Profile**
- Geography: **Statistical Area Level 1**
- State: **WA**

Unzip and find `2021Census_G01_WA_SA1.csv` in the `2021 Census GCP ... SA1 for WA`
folder. Table **G01** is *Selected Person Characteristics by Sex*; the script
reads the `SA1_CODE_2021` and `Tot_P_P` (total persons) columns.

This is population by **place of usual residence** — where people live. That is
the right basis for the origin side of a stop catchment.

## 3. Employment — required

**Get:** `2021Census_W01B_WA_POW_SA2.csv`

From [Census DataPacks](https://www.abs.gov.au/census/find-census-data/datapacks):

- DataPack type: **Working Population Profile**
- Geography: **Statistical Area Level 2**
- State: **WA**

Two things that are easy to get wrong here:

**W01 arrives split as W01A and W01B.** DataPacks split wide tables across files.
The age-by-sex breakdown runs through `W01A` and spills into `W01B`; the persons
totals live *only* in `W01B`. The script reads **W01B only** and ignores W01A
entirely. Keep W01A if you later want the age breakdown, but nothing reads it now.

**The code column is `POW_SA2_CODE_2021`, not `SA2_CODE_2021`.** Place-of-work
geography is its own code set. It also contains non-spatial special codes —
`597979799` (no fixed address) and `599999499` (undefined) — which have no
geometry and drop out on the join. The value column is `P_Tot_Tot`: total
employed persons counted at that place of work, i.e. jobs.

Greater Perth totals **972,121 jobs** across 185 SA2s.

> **Why this is not optional.** G01 `Tot_P_P` is place of usual *residence*: it
> counts who sleeps in an SA1 and says nothing about jobs located there. Measured
> on your top 200 stops, jobs (mean 2,089 per 400 m catchment) outweigh residents
> (mean 1,112). Wellington St Stand 10 has 1,323 residents against 14,755 jobs.
> Dropping employment moves the average stop 34 rank places out of 200, and moves
> 34 stops by more than 50 places.

> **Why SA2 and not SA1.** The Working Population Profile is counted by place of
> work, and the ABS does not publish it below SA2. Population therefore comes from
> small SA1 polygons while jobs come from much coarser SA2 polygons. This has a
> visible consequence: because a 400 m catchment sits entirely inside a CBD SA2,
> all 13 central stops in the test run received an identical job count of 14,754.7.
> Employment cannot currently discriminate between stops within one SA2. See
> "Mesh blocks" below.

---

## Mesh blocks — dasymetric apportionment (`--dasymetric`)

The GeoPackage you already have includes `MB_2021_AUST_GDA2020`: 368,286 mesh
blocks, each carrying `MB_CATEGORY_2021` (Residential, Commercial, Industrial,
Parkland, Education, Hospital, Transport, Water, ...).

The script currently apportions by **area weighting** — a catchment covering 30%
of a zone's area gets 30% of its count, which assumes people and jobs are spread
uniformly across the zone. They are not. Mesh block categories let you do
**dasymetric** apportionment instead: spread SA1 population only across the
residential mesh blocks inside it, and SA2 employment only across the commercial
and industrial ones.

**This is implemented.** Pass `--dasymetric`. Greater Perth has 27,112 mesh
blocks covering all 4,822 SA1s and all 185 SA2s, so every zone is refinable.
Mesh blocks carry `SA1_CODE_2021` and `SA2_CODE_2021` directly, so zones and
mesh blocks are matched by attribute join — no spatial operation, no slivers.

Land-use weights live in `POP_MB_WEIGHTS` and `JOB_MB_WEIGHTS` at the top of the
script. They are normalised within each zone, so **zone totals are conserved
exactly** whatever weights you choose; the script verifies this each run and
refuses to continue if a surface loses count. These weights are knowledge-driven
parameters in the Wu et al. sense — vary them and report the sensitivity.

Measured effect on the top 200 stops, radial catchments:

| | Area weighted | Dasymetric |
|---|---|---|
| Mean `A_i` | 3,202 | 3,731 |
| Distinct job values (of 200 stops) | 141 | 196 |
| Mean absolute rank shift | — | 24.4 places |
| Stops moving >50 places | — | 27 |

The jump from 141 to 196 distinct job values is the CBD tie breaking. Wellington
St and St Georges Tce stops gain 4,000–8,200 jobs each because they sit on dense
commercial blocks; stops near Kings Park and the Supreme Court Gardens lose,
because area weighting had been crediting them with parkland as though it held
workplaces.

Two behaviours worth knowing. 174 SA1s (3.6%) have residents but no residential
mesh block — the script falls back to area weighting *within* those zones rather
than letting their population vanish. And under dasymetric weighting a stop
surrounded entirely by parkland or commercial land can legitimately reach zero
resident population, where area weighting would have given it a spurious share.

Mesh block *counts* are a separate small download (2021 Census Mesh Block counts)
if you ever want population at MB resolution directly — but the category field
alone, which you already have, is what drives the redistribution.

Outputs: `apportionment_summary.txt` and `apportionment_comparison.csv`.

---

## The band half-width — read this before quoting any radial-vs-network number

The network catchment is built by tracing 400 m along the walk graph and drawing
a band of half-width `EDGE_BUFFER_M` around the reachable streets. **That
half-width alone determines how much smaller a network catchment looks than a
radial one**, so it governs the headline result:

| Half-width | % of the 400 m disc | Implied "radial overstates by" |
|---|---|---|
| 20 m | ~24% | ~4.2× |
| **40 m (in use)** | ~49% | ~2.0× |
| 75 m | ~93% | ~1.1× |
| 150 m | ~193% | ~0.5× |

Turning one dial moves the claim from "radial overstates fourfold" to "network
catchments are larger than radial ones". A first run at 25 m produced a median
ratio of 3.95 and "99.8% of stops overstated" — that was the parameter talking,
not severance.

40 m is set from Perth's suburban block geometry: lots run roughly 30–40 m deep
with about 70–80 m between parallel streets, so a 40 m half-width picks up the
parcels fronting a street without reaching across the block into the next one.

Every run writes `catchment_sensitivity.csv` and prints the sweep into
`catchment_summary.txt`. **Report that table alongside any radial-vs-network
claim.** The defensible finding is not the absolute ratio — it is the *variation*
in the ratio between stops at a fixed half-width, which is what actually measures
severance.

## Which A_i to use in the index

The script emits `A_pop`, `A_job`, `A` and `A_z` **without a method suffix**.
That is the canonical one: network catchment everywhere it is usable, radial
substituted where the walk graph could not support one. `A_source` records which
was used per stop. **Use `A_z` when you add the component to the SII.**

The suffixed columns (`A_network`, `A_radial`, …) are retained so the
substitution is auditable, not because both are meant to be reported.

Run with `--method both` even though network is the operative method — radial
has to exist for the fallback to draw on.

## Suspect catchments

Stops whose snapped node lands on a sparse or disconnected fragment of the walk
graph produce near-empty catchments — ratios in the thousands that read as
severance but are data defects. In the 25 m run, 154 stops returned less than one
person and headed the "most overstated" table: Spearwood Ave, Rockingham Rd,
Leach Hwy, Murdoch Dr, all high-speed roads where OSM footpath coverage is thin.

Stops are now flagged `catchment_suspect` when total reachable street length is
under `MIN_REACH_M` (400 m) or the snap distance exceeds `MAX_SNAP_M` (100 m).
They are excluded from the comparison and the count is reported. The flag,
`n_reachable`, `reach_len_m` and `snap_dist_m` are all carried into the output
CSV, so you can inspect them in QGIS rather than take the exclusion on trust.

## Python packages

```bash
pip install geopandas osmnx
```

`geopandas` is required. `osmnx` is only needed for the network catchment stage —
`--method radial` runs without it.

## Running it

```bash
# smoke test, seconds — verified working on your files
python sii_catchment_population.py --method radial --limit 200 --dasymetric

# full run
python sii_catchment_population.py --method both --dasymetric --export-polygons
```

Start with the smoke test. Once the numbers look sane, run `--method both`.

**Expect the first full run to be slow.** It downloads the Perth walk network
from OpenStreetMap (once — it is cached to `data/cache/perth_walk.graphml`, so
later runs skip it), then traces a bounded Dijkstra for each of 11,750 stops.
Budget the better part of an hour. Progress prints every 500 stops.

## What comes out

| File | Contents |
|---|---|
| `output/sii_bus_full_with_catchment.csv` | your SII table plus `A_pop_*`, `A_job_*`, `A_*`, `A_*_z`, `catchment_area_*` |
| `output/catchment_comparison.csv` | per-stop radial vs network, with ratios |
| `output/catchment_summary.txt` | overstatement stats, the sensitivity table, worst-affected stops |
| `output/catchment_sensitivity.csv` | median catchment area at each band half-width |
| `output/apportionment_comparison.csv` | per-stop dasymetric vs area weighting (`--dasymetric`) |
| `output/apportionment_summary.txt` | rank shifts and the largest gains/losses (`--dasymetric`) |
| `output/catchment_comparison.png` | three diagnostic panels |
| `output/geojson/catchments_*.gpkg` | catchment polygons for QGIS (`--export-polygons`) |

By default the script writes a **new** CSV and leaves `sii_bus_full.csv` alone.
Pass `--in-place` to merge the columns into `sii_bus_full.csv` instead; it keeps
a `.bak` of the original the first time.

## What it deliberately does not do

It computes and standardises `A_i` but does **not** add it to the SII. The weight
vector in `sii_bus_stops_pipeline.py` is untouched, because introducing a seventh
component means re-deciding all seven weights so they still sum to 1 — a
methodological choice, not a mechanical one.

The `catchment_summary.txt` overstatement figures are the thing to read first.
They are the evidence for the radial-versus-network claim the literature review
commits to, and they stand as a result in their own right regardless of what you
decide about the weights.
