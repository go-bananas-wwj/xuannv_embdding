# P3A.0 OSM Weak Label Audit

- Generated at: 2026-06-29T15:10:18.834900+00:00
- Output root: `/data/xuannv_embedding/experiments/p3a_osm_audit_20260629`
- Scope: current OSM caches and raster masks for Haidian/Harbin building and road labels.
- Rule: these labels are weak auxiliary signals only; OSM blanks are not strong negatives.

## Executive Summary

Some current caches contain sparse OSM time tags. They can provide confidence hints, but not month-level ground truth change labels.

## Cache and Mask Coverage

| region | task | features | geometry | time columns with values | masks | positive patches | mean positive pixels | p95 positive pixels |
|---|---:|---:|---|---|---:|---:|---:|---:|
| haidian | building_osm | 44484 | MultiPolygon:26, Point:108, Polygon:44350 | start_date, opening_date, construction_date, check_date, disused:amenity | 320 | 290 (90.62%) | 9.60% | 33.09% |
| haidian | road_osm | 77675 | GeometryCollection:1, LineString:60607, MultiPolygon:3, Point:16318, Polygon:746 | - | 320 | 317 (99.06%) | 15.63% | 27.62% |
| harbin | building_osm | 13093 | MultiPolygon:2, Point:7, Polygon:13084 | - | 424 | 166 (39.15%) | 3.02% | 14.94% |
| harbin | road_osm | 10523 | LineString:9894, Point:571, Polygon:58 | - | 424 | 375 (88.44%) | 7.23% | 21.69% |

## Temporal Tag Detail

### haidian / building_osm

| column | non-null | ratio | examples |
|---|---:|---:|---|
| `start_date` | 45 | 0.10% | 1958-04-22, 1997-01-01, 1771, 1772, 2023-09-28, 1758, 2003竣工, 2000审批 |
| `opening_date` | 1 | 0.00% | 2025-10-18 |
| `construction_date` | 11 | 0.02% | 1930, 1435, 1924, 1644, 1436, 1576, 1891, 1941 |
| `check_date` | 12 | 0.03% | 2026-04-15, 2026-04-15, 2025-03-10, 2025-11-02, 2024-04-22, 2024-04-14, 2025-08-01, 2021-06-17 |
| `disused:amenity` | 2 | 0.00% | toilets, post_office |

### haidian / road_osm

- No configured time/lifecycle columns are present.

### harbin / building_osm

- No configured time/lifecycle columns are present.

### harbin / road_osm

- No configured time/lifecycle columns are present.

## Semantic Tag Detail

### haidian / building_osm
- `building` non-null 44484: yes:31648, residential:4220, apartments:2396, bungalow:1244, house:896, school:827, commercial:540, university:490, office:295, dormitory:254, wall:179, industrial:158
- `building:use` non-null 4: service:1, retail:1, parking:1, public:1
- `construction` non-null 2: yes:2
- `landuse` non-null 11: military:6, commercial:2, retail:2, research:1
- `amenity` non-null 1010: restaurant:149, toilets:137, place_of_worship:130, shelter:116, bicycle_parking:86, fuel:77, library:45, police:31, theatre:28, fast_food:21, cafe:16, community_centre:15
- `man_made` non-null 114: tower:73, works:11, storage_tank:8, gasometer:5, cooling:5, cooling_tower:4, water_well:2, observatory:2, pumping_station:2, wastewater_plant:1, pier:1

### haidian / road_osm
- `highway` non-null 77675: footway:17129, service:10044, residential:7650, crossing:6550, bus_stop:4825, tertiary:4606, path:4586, traffic_signals:4196, unclassified:2211, secondary:2197, cycleway:2012, steps:1920

### harbin / building_osm
- `building` non-null 13093: yes:11146, apartments:952, residential:268, industrial:221, house:96, university:86, commercial:63, retail:48, dormitory:37, warehouse:34, school:26, college:20

### harbin / road_osm
- `highway` non-null 10523: service:1993, footway:1607, tertiary:1122, residential:977, secondary:850, path:652, trunk:600, unclassified:546, primary:442, trunk_link:316, crossing:240, bus_stop:223

## Training Decision

1. Use current OSM building/road only as weak density or presence priors.
2. Do not use current OSM as monthly change labels.
3. Do not treat empty OSM areas as hard background negatives.
4. For P3A monthly training, prefer density targets and confidence masks over hard masks.
5. To use true OSM time information, download full-history `.osh.pbf` or monthly snapshots, then rerun this audit on history-derived GeoPackage layers.

## Next Step

Build P3A.1 index-reconstruction targets first, while preparing an OSM-history ingestion path in parallel. This avoids blocking the embedding upgrade on sparse OSM history.
