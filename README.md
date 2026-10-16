# A Spatial Analysis of Perth's Transport Hubs

Code accompanying the honours dissertation *A Spatial Analysis of Perth's
Transport Hubs: How urban form affects the robustness of public transport*
(Ganesh Chinnasamy, supervised by Dr Nur Halder).

The pipeline builds a stop-level diagnostic for Perth's scheduled public
transport network from openly available data: a **Stop Influence Index**
combining service frequency, route diversity, transfer potential, structural
centrality and catchment demand, and a **Hub Reliance Index** that amplifies it
by the degree to which a catchment has no practical alternative.

Interactive results: <https://ganeshcy.github.io/Perth_HRI_Visualisation_Tool/>

---

## What is and is not in this repository

**Included.** The current analysis pipeline — everything in
`merged_node_variant/` — together with the master results table
`merged_node_variant/output/node_master.csv`. That table is committed
deliberately: it carries one row per scored node with every component,
intermediate and derived value, so every table and figure in the dissertation
can be checked without reproducing the full pipeline.

**Not included.** None of the source data, and none of the large generated
intermediates. Together they come to roughly 4.5 GB, and several individual
files exceed GitHub's 100 MB limit — the cached pedestrian network alone is
414 MB. The sources are all public; see *Obtaining the data* below.

Also not included: the per-stand baseline scripts that sat at the root of this
directory. Chapter 4 reports baseline results and §4.10 compares the two
representations, so those scripts exist and were used, but they are superseded
and several carry absolute paths from an earlier machine. The repository
publishes the line of work the dissertation's current figures come from.

---

## Layout

The repository root corresponds to `Research Proj/`. The pipeline resolves the
GTFS feed as a sibling of `merged_node_variant/`, so the data directories go at
the root alongside it, exactly as laid out below.

```
<repo root>
  google_transit/                   (you provide — see Obtaining the data)
  Spatial_Data/                     (you provide)
  data/abs/                         (you provide)
  data/cache/                       (created on first run; 414 MB walk graph)
  merged_node_variant/              THE PIPELINE
    walk_snap.py                    edge-based snapping of stops to the walk graph
    sii_bus_stops_pipeline_merged.py    F, R, T, C
    sii_catchment_population_merged.py  catchment demand D
    sii_v_catchment.py              car dependence V
    sii_alternative_access_merged.py    alternative access A
    sii_transfer_potential_merged.py    transfer potential T
    sii_congestion_merged.py        scheduled congestion exposure G
    sii_critic_weights.py           derives the component weights
    sii_assemble.py                 joins everything; writes node_master.csv
    sii_export_merged.py            GeoJSON + web payload
    build_map.py / build_map_leaflet.py  the two editions of the explorer
    output/node_master.csv          the master results table
  README.md, requirements.txt
```

A node here is a facility rather than a kerb: stops sharing a GTFS
`parent_station` are collapsed to a single node, so a station or interchange
appears once rather than once per stand. §3.3.2 of the dissertation sets out
why, and §4.10 reports what the choice changes.

---

## Obtaining the data

| Source | Where | Notes |
|---|---|---|
| Transperth GTFS | Transperth / data.wa.gov.au | A pinned extract was used, service period 3 March – 1 June 2026. The feed is republished nightly, so a fresh download will **not** reproduce the reported figures exactly. |
| PTA stops and services | data.wa.gov.au (PTA-001, PTA-002) | Shapefiles. |
| ASGS Edition 3 boundaries | ABS | Mesh block, SA1, SA2, GDA2020. Edition 3 specifically — it is the vintage the 2021 Census was coded to. |
| Census 2021 tables G01, W01B, G34 | ABS | Population at SA1, employment at SA2, motor vehicles at SA1. |
| Pedestrian network | OpenStreetMap, via OSMnx | Downloaded and cached automatically on first run. |

Place them as `google_transit/`, `Spatial_Data/` and `data/abs/` relative to
`Research Proj/`. The walk graph caches itself to `data/cache/`.

Each source carries its own licence, and none is redistributed here. Note in
particular that anything derived from OpenStreetMap — the walk catchments and
everything computed from them — falls under the ODbL, whose share-alike
provisions may apply to redistribution of derived data. The code in this
repository is licensed separately; see `LICENSE`.

---

## Running it

```bash
pip install -r requirements.txt
cd merged_node_variant
```

Stage order matters, because each stage consumes files the previous one writes:

```bash
python sii_transfer_potential_merged.py --thresholds 500 --adopt 500
python sii_bus_stops_pipeline_merged.py
python sii_catchment_population_merged.py --dasymetric --export-polygons \
                                          --method both --allocation supply
python sii_v_catchment.py --metres 800 --simple --fallback
python sii_alternative_access_merged.py --opposing-radius 200
python sii_critic_weights.py --standardise plain --t-spatial 500
python sii_assemble.py
python sii_export_merged.py && python build_map_leaflet.py
```

Two flags are easy to omit and fail silently if you do. Without
`--dasymetric`, catchment demand is apportioned by plain area weighting rather
than by land use. Without `--export-polygons`, the catchment GeoPackages are
not rewritten, and the alternative-access stage then reads whichever vintage is
on disk. Both are mistakes that were made during development and took a while
to notice, because the output looks entirely reasonable either way.

Expect the first run to take some hours: the catchment stages perform one
bounded Dijkstra per stop over a graph of 342,563 nodes and 997,890 edges, and
the walk graph has to be downloaded once.

---

## Known limitations

These are set out in the Limitations section of the dissertation and are not
restated here, so that there is one authoritative list rather than two that can
drift apart. In summary: the instrument is not validated against realised
behaviour, because no stop-level patronage data for Perth is public; catchment
demand and car dependence are measured over different catchment radii, which
biases the captive class toward stops serving workplaces rather than
residents; alternative access credits an adjacent stop with its full service
without regard to whether it carries the same route; the weighting procedure
rewards dispersion; the reliance amplification is declared rather than
estimated; and the analysis rests on a single pinned GTFS extract covering one
thirteen-week service period.
