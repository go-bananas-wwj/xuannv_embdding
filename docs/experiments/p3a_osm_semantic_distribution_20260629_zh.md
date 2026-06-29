# P3A OSM 语义分布调研

- 生成时间：2026-06-29T15:26:50.397737+00:00
- 数据输出：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629`
- 目的：把 OSM 作为静态/半静态弱语义先验，调研哪些标签能服务月度 embedding 训练。
- 约束：OSM 标签不作为强真值；OSM 空白不作为强负样本；下游人工标注不进入主 embedding 训练。

## 总体建议

OSM 可以用于训练大概语义分布，尤其是建筑、道路、居住/商业/工业/农业/水体/绿地等粗粒度类别。
建议使用密度图、面积比例图、距离图或 confidence mask，而不是逐像素硬标签。

## 区域级语义覆盖

### haidian

| 弱语义组 | 要素数 | 面积 km2 | 线长度 km | 建议用途 |
|---|---:|---:|---:|---|
| minor_road | 20546 | 0.04 | 4793.26 | 道路/交通线密度和距离图 |
| path_walk | 27644 | 1.13 | 3852.20 | 低权重语义先验 |
| major_road | 9958 | 0.00 | 3771.23 | 道路/交通线密度和距离图 |
| forest_green | 3565 | 324.49 | 61.52 | 土地利用面积比例/密度弱监督 |
| rail_transport | 2722 | 0.19 | 1289.73 | 道路/交通线密度和距离图 |
| residential | 2408 | 117.31 | 0.00 | 土地利用面积比例/密度弱监督 |
| culture_recreation | 5016 | 104.63 | 3.18 | 功能区 POI 密度弱监督 |
| water_wetland | 1928 | 39.35 | 570.55 | 水体/湿地密度弱监督 |
| education_public | 2752 | 55.29 | 0.00 | 功能区 POI 密度弱监督 |
| military | 218 | 51.57 | 0.00 | 低权重语义先验 |
| agriculture | 959 | 43.28 | 0.00 | 土地利用面积比例/密度弱监督 |
| industrial | 873 | 38.37 | 0.00 | 土地利用面积比例/密度弱监督 |
| commercial | 782 | 36.12 | 0.00 | 土地利用面积比例/密度弱监督 |
| generic_building | 33509 | 35.85 | 0.00 | 建筑密度和边界弱监督 |
| construction | 643 | 22.56 | 0.00 | 低权重语义先验 |
| transport | 2085 | 10.84 | 7.05 | 道路/交通线密度和距离图 |
| commercial_service | 6437 | 8.23 | 0.00 | 功能区 POI 密度弱监督 |
| residential_building | 7866 | 6.88 | 0.00 | 建筑密度和边界弱监督 |
| health_public | 310 | 4.74 | 0.00 | 功能区 POI 密度弱监督 |
| commercial_building | 1082 | 3.21 | 0.00 | 建筑密度和边界弱监督 |
| infrastructure | 765 | 1.92 | 3.33 | 低权重语义先验 |
| cemetery | 22 | 1.59 | 0.00 | 低权重语义先验 |
| industrial_building | 238 | 0.47 | 0.00 | 建筑密度和边界弱监督 |
| public_service | 1015 | 0.36 | 0.00 | 低权重语义先验 |
| bare_sparse | 84 | 0.06 | 0.00 | 低权重语义先验 |
| road_poi | 15856 | 0.00 | 0.00 | 道路/交通线密度和距离图 |

### harbin

| 弱语义组 | 要素数 | 面积 km2 | 线长度 km | 建议用途 |
|---|---:|---:|---:|---|
| water_wetland | 692 | 249.00 | 281.85 | 水体/湿地密度弱监督 |
| major_road | 3170 | 0.00 | 2469.13 | 道路/交通线密度和距离图 |
| minor_road | 3667 | 0.00 | 1278.58 | 道路/交通线密度和距离图 |
| residential | 1278 | 93.13 | 0.00 | 土地利用面积比例/密度弱监督 |
| rail_transport | 1141 | 0.00 | 726.24 | 道路/交通线密度和距离图 |
| forest_green | 944 | 42.76 | 6.39 | 土地利用面积比例/密度弱监督 |
| path_walk | 2517 | 0.62 | 412.28 | 低权重语义先验 |
| industrial | 551 | 41.71 | 0.00 | 土地利用面积比例/密度弱监督 |
| culture_recreation | 1094 | 31.47 | 3.60 | 功能区 POI 密度弱监督 |
| generic_building | 11392 | 18.46 | 0.00 | 建筑密度和边界弱监督 |
| education_public | 361 | 14.97 | 0.00 | 功能区 POI 密度弱监督 |
| commercial | 343 | 11.76 | 0.00 | 土地利用面积比例/密度弱监督 |
| construction | 158 | 11.31 | 0.00 | 低权重语义先验 |
| agriculture | 59 | 8.30 | 0.00 | 土地利用面积比例/密度弱监督 |
| transport | 352 | 5.02 | 0.00 | 道路/交通线密度和距离图 |
| commercial_service | 894 | 2.87 | 0.00 | 功能区 POI 密度弱监督 |
| infrastructure | 305 | 0.52 | 11.65 | 低权重语义先验 |
| health_public | 61 | 1.60 | 0.00 | 功能区 POI 密度弱监督 |
| industrial_building | 255 | 1.59 | 0.00 | 建筑密度和边界弱监督 |
| residential_building | 1353 | 1.40 | 0.00 | 建筑密度和边界弱监督 |
| military | 9 | 0.77 | 0.00 | 低权重语义先验 |
| commercial_building | 130 | 0.43 | 0.00 | 建筑密度和边界弱监督 |
| public_service | 71 | 0.26 | 0.00 | 低权重语义先验 |
| cemetery | 1 | 0.05 | 0.00 | 低权重语义先验 |
| bare_sparse | 69 | 0.02 | 0.00 | 低权重语义先验 |
| road_poi | 521 | 0.00 | 0.00 | 道路/交通线密度和距离图 |

## 主题明细

### haidian / landuse

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/haidian_landuse.gpkg`
- 要素数：8348
- 几何类型：{"Polygon": 8254, "Point": 57, "MultiPolygon": 36, "LineString": 1}

- `amenity` 有值 94；Top values：research_institute:41, place_of_worship:16, hospital:13, driver_training:7, university:4, college:3, studio:2, parking:2, courthouse:1, police:1, food_court:1, grave_yard:1
- `building` 有值 11；Top values：yes:8, retail:2, college:1
- `highway` 有值 1；Top values：pedestrian:1
- `landuse` 有值 8348；Top values：residential:2401, forest:1539, grass:864, construction:610, commercial:554, brownfield:452, farmland:398, orchard:397, industrial:290, military:178, retail:161, farmyard:79
- `leisure` 有值 27；Top values：park:24, garden:2, stadium:1
- `man_made` 有值 1；Top values：wastewater_plant:1
- `natural` 有值 8；Top values：grassland:5, scrub:2, shrubbery:1
- `office` 有值 14；Top values：government:10, company:1, political_party:1, yes:1, research:1
- `shop` 有值 3；Top values：mall:1, car:1, wholesale:1
- `tourism` 有值 25；Top values：attraction:7, yes:7, hotel:6, museum:5

### haidian / natural

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/haidian_natural.gpkg`
- 要素数：6039
- 几何类型：{"Point": 2832, "LineString": 1857, "Polygon": 1334, "MultiPolygon": 11, "MultiLineString": 5}

- `amenity` 有值 1；Top values：fountain:1
- `highway` 有值 13；Top values：path:12, track:1
- `landuse` 有值 8；Top values：meadow:5, grass:2, construction:1
- `leisure` 有值 6；Top values：park:6
- `natural` 有值 5189；Top values：tree:2360, water:777, tree_row:586, cliff:528, wood:185, scrub:160, peak:148, sand:78, rock:78, grassland:68, spring:53, cave_entrance:44
- `tourism` 有值 153；Top values：attraction:153
- `water` 有值 425；Top values：pond:137, river:109, lake:90, basin:50, reservoir:19, canal:16, reflecting_pool:1, moat:1, well:1, stream:1
- `waterway` 有值 849；Top values：river:245, drain:193, canal:122, stream:108, weir:60, dam:53, ditch:50, waterfall:14, lock_gate:4

### haidian / transport

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/haidian_transport.gpkg`
- 要素数：84495
- 几何类型：{"LineString": 63082, "Point": 20003, "Polygon": 853, "MultiLineString": 554, "MultiPolygon": 3}

- `amenity` 有值 2；Top values：bank:1, 捷运:1
- `building` 有值 40；Top values：yes:22, hangar:15, train_station:2, transportation:1
- `highway` 有值 77748；Top values：footway:17142, service:10041, residential:7645, crossing:6554, bus_stop:4824, tertiary:4609, path:4586, traffic_signals:4244, unclassified:2208, secondary:2191, cycleway:2035, steps:1920
- `landuse` 有值 3；Top values：military:2, recreation_ground:1
- `leisure` 有值 4；Top values：slipway:4
- `man_made` 有值 4；Top values：pier:2, bridge:1, tower:1
- `natural` 有值 12；Top values：ridge:5, valley:5, earth_bank:2
- `railway` 有值 6685；Top values：rail:1260, subway:1183, switch:1076, level_crossing:770, buffer_stop:538, subway_entrance:498, stop:428, platform:248, station:206, disused:83, razed:70, monorail:64
- `tourism` 有值 6；Top values：attraction:6
- `waterway` 有值 1；Top values：dam:1

### haidian / built

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/haidian_built.gpkg`
- 要素数：46146
- 几何类型：{"Polygon": 44730, "Point": 1191, "LineString": 195, "MultiPolygon": 30}

- `amenity` 有值 1011；Top values：restaurant:149, toilets:137, place_of_worship:130, shelter:116, bicycle_parking:86, fuel:77, library:45, police:31, theatre:28, fast_food:21, cafe:16, community_centre:15
- `building` 有值 44489；Top values：yes:31649, residential:4220, apartments:2396, bungalow:1248, house:896, school:827, commercial:540, university:490, office:295, dormitory:254, wall:179, industrial:158
- `highway` 有值 149；Top values：construction:121, tertiary:9, secondary:7, residential:4, service:3, footway:2, unclassified:1, disused:1, pedestrian:1
- `landuse` 有值 29；Top values：construction:15, military:6, commercial:2, retail:2, residential:2, research:1, industrial:1
- `leisure` 有值 93；Top values：sports_hall:29, sports_centre:22, fitness_centre:17, stadium:14, bleachers:2, track:2, swimming_pool:1, dance:1, ice_rink:1, pitch:1, park:1, bandstand:1
- `man_made` 有值 1604；Top values：surveillance:473, tower:405, bridge:275, pier:64, flagpole:63, antenna:62, mast:45, chimney:34, water_well:20, storage_tank:17, planter:16, ceremonial_gate:16
- `office` 有值 279；Top values：research:91, government:68, yes:42, company:23, educational_institution:14, it:11, political_party:9, security:5, water_utility:3, union:3, publisher:2, telecommunication:2
- `railway` 有值 26；Top values：station:12, workshop:7, signal_box:2, construction:1, platform:1, radio:1, car_wash:1, wash:1
- `shop` 有值 216；Top values：mall:57, supermarket:34, yes:21, car:17, convenience:12, ticket:12, department_store:10, books:10, gift:8, florist:4, furniture:3, wholesale:3
- `tourism` 有值 233；Top values：hotel:103, museum:58, attraction:41, artwork:12, information:7, guest_house:5, gallery:4, aquarium:1, hostel:1, apartment:1

### haidian / activity

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/haidian_activity.gpkg`
- 要素数：17521
- 几何类型：{"Point": 8877, "Polygon": 8504, "LineString": 99, "MultiPolygon": 25, "MultiLineString": 16}

- `amenity` 有值 9378；Top values：parking:1498, restaurant:1448, toilets:639, school:616, bank:436, bench:434, fast_food:392, cafe:364, shelter:317, bicycle_parking:295, place_of_worship:229, kindergarten:216
- `building` 有值 1814；Top values：yes:1665, commercial:29, roof:20, school:15, office:10, university:10, house:9, retail:8, hotel:7, stadium:7, bungalow:6, parking:5
- `highway` 有值 10；Top values：service:4, unclassified:2, residential:2, pedestrian:2
- `landuse` 有值 177；Top values：commercial:59, military:32, grass:20, religious:17, industrial:15, farmland:7, forest:7, retail:4, education:3, construction:2, residential:2, religional:1
- `leisure` 有值 4123；Top values：pitch:2030, park:1023, garden:298, track:231, stadium:91, sports_centre:86, fitness_centre:78, playground:47, fitness_station:46, sports_hall:33, swimming_pool:27, outdoor_seating:19
- `man_made` 有值 15；Top values：water_well:8, tower:4, ceremonial_gate:2, bridge:1
- `natural` 有值 159；Top values：rock:65, cliff:61, tree:9, valley:6, wood:5, grass:5, cave_entrance:2, peak:1, fossil:1, spring:1, saddle:1, stone:1
- `office` 有值 701；Top values：government:149, research:138, company:118, yes:88, educational_institution:39, estate_agent:33, telecommunication:29, quango:18, it:14, political_party:13, ngo:8, security:8
- `railway` 有值 2；Top values：subway_entrance:2
- `shop` 有值 1606；Top values：convenience:261, supermarket:241, bakery:87, clothes:84, mall:78, hairdresser:67, books:54, yes:49, kiosk:45, ticket:39, greengrocer:35, gift:33
- `tourism` 有值 1757；Top values：attraction:533, hotel:391, artwork:303, viewpoint:124, museum:108, information:83, chalet:81, motel:26, hostel:26, picnic_site:20, guest_house:12, gallery:11
- `water` 有值 1；Top values：well:1

### harbin / landuse

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/harbin_landuse.gpkg`
- 要素数：2860
- 几何类型：{"Polygon": 2855, "MultiPolygon": 5}

- `highway` 有值 1；Top values：pedestrian:1
- `landuse` 有值 2860；Top values：residential:1264, industrial:417, commercial:322, forest:230, grass:198, construction:149, meadow:64, governmental:44, railway:36, farmland:23, retail:20, proposed:17
- `leisure` 有值 21；Top values：pitch:21
- `office` 有值 1；Top values：government:1
- `tourism` 有值 2；Top values：attraction:1, museum:1

### harbin / natural

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/harbin_natural.gpkg`
- 要素数：1242
- 几何类型：{"Polygon": 933, "LineString": 176, "Point": 127, "MultiPolygon": 5, "MultiLineString": 1}

- `natural` 有值 1135；Top values：water:433, wood:218, scrub:124, tree:123, sand:69, tree_row:59, wetland:55, grassland:28, beach:12, shrubbery:11, hot_spring:2, peak:1
- `water` 有值 99；Top values：pond:47, river:24, lake:21, basin:5, wastewater:1, oxbow:1
- `waterway` 有值 107；Top values：drain:39, river:30, canal:20, stream:10, yes:4, ditch:4

### harbin / transport

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/harbin_transport.gpkg`
- 要素数：13217
- 几何类型：{"LineString": 10978, "Point": 2084, "MultiLineString": 92, "Polygon": 63}

- `building` 有值 1；Top values：yes:1
- `highway` 有值 10523；Top values：service:1993, footway:1607, tertiary:1122, residential:977, secondary:850, path:652, trunk:600, unclassified:546, primary:442, trunk_link:316, crossing:240, bus_stop:223
- `landuse` 有值 1；Top values：tourism:1
- `man_made` 有值 2；Top values：tower:2
- `railway` 有值 2694；Top values：rail:992, switch:740, buffer_stop:246, level_crossing:199, subway:100, stop:75, subway_entrance:70, station:49, platform:48, milestone:46, crossing:42, railway_crossing:27
- `tourism` 有值 3；Top values：attraction:3

### harbin / built

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/harbin_built.gpkg`
- 要素数：13662
- 几何类型：{"Polygon": 13219, "Point": 265, "LineString": 170, "MultiPolygon": 8}

- `amenity` 有值 90；Top values：restaurant:18, toilets:9, fuel:8, place_of_worship:7, library:7, theatre:6, post_office:5, police:5, courthouse:3, community_centre:2, cinema:2, townhall:2
- `building` 有值 13093；Top values：yes:11146, apartments:952, residential:268, industrial:221, house:96, university:86, commercial:63, retail:48, dormitory:37, warehouse:34, school:26, college:20
- `highway` 有值 33；Top values：construction:33
- `landuse` 有值 20；Top values：residential:14, construction:5, industrial:1
- `leisure` 有值 13；Top values：sports_centre:6, sports_hall:5, golf_course:1, ice_rink:1
- `man_made` 有值 530；Top values：pier:148, tower:96, mast:71, chimney:66, bridge:56, storage_tank:28, works:14, antenna:9, flagpole:8, wastewater_plant:5, embankment:4, chiller:4
- `natural` 有值 2；Top values：water:2
- `office` 有值 43；Top values：government:29, educational_institution:4, yes:4, research:2, company:1, energy_supplier:1, property_management:1, telecommunication:1
- `railway` 有值 3；Top values：radio:2, signal_box:1
- `shop` 有值 61；Top values：mall:34, car:12, department_store:3, houseware:2, ticket:2, yes:2, convenience:1, electronics:1, telecommunication:1, funeral_directors:1, shoes:1, wholesale:1
- `tourism` 有值 35；Top values：hotel:15, museum:9, yes:3, attraction:2, gallery:2, aquarium:1, zoo:1, viewpoint:1, information:1
- `water` 有值 2；Top values：river:2

### harbin / activity

- 缓存：`/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/harbin_activity.gpkg`
- 要素数：2747
- 几何类型：{"Polygon": 1696, "Point": 1034, "LineString": 11, "MultiPolygon": 4, "MultiLineString": 2}

- `amenity` 有值 1140；Top values：parking:292, school:140, restaurant:139, cafe:78, hospital:52, shelter:51, fuel:41, fast_food:36, bank:35, toilets:27, university:25, college:16
- `building` 有值 238；Top values：yes:213, roof:8, retail:7, commercial:2, hotel:1, church:1, cathedral:1, construction:1, mosque:1, office:1, toilets:1, school:1
- `highway` 有值 3；Top values：pedestrian:2, footway:1
- `landuse` 有值 30；Top values：grass:21, industrial:5, tourism:2, railway:1, commercial:1
- `leisure` 有值 971；Top values：park:475, pitch:259, garden:119, stadium:42, track:18, sports_centre:12, swimming_pool:10, playground:6, sports_hall:5, sauna:5, ice_rink:4, resort:4
- `man_made` 有值 2；Top values：tower:2
- `office` 有值 57；Top values：government:34, yes:7, company:4, educational_institution:4, research:3, publisher:1, telecommunication:1, estate_agent:1, energy_supplier:1, property_management:1
- `shop` 有值 249；Top values：mall:41, clothes:34, convenience:25, supermarket:22, car:14, department_store:8, bakery:5, hairdresser:5, deli:5, variety_store:5, sports:5, mobile_phone:5
- `tourism` 有值 329；Top values：hotel:91, information:76, artwork:58, attraction:48, museum:29, theme_park:7, hostel:5, zoo:5, yes:3, gallery:2, guest_house:2, aquarium:1

## 可进入 P3A 的 OSM 弱标签

优先级高：

1. `building`：建筑密度、建筑面积比例、建筑边界/距离图。
2. `highway`：主路/小路/步行路分层道路密度、道路中心线距离图。
3. `landuse=residential/commercial/industrial/farmland/forest/grass`：粗土地利用分布。
4. `natural=water/wood/wetland/grassland` 和 `waterway=*`：水体、湿地、绿地弱监督。

优先级中：

1. `amenity/shop/office/tourism/leisure`：功能区 POI 密度，例如商业服务、教育公共、医疗公共、文体休闲。
2. `man_made`：工业设施和基础设施分布。
3. `railway/aeroway`：交通设施先验。

暂时不用作强监督：

1. 过细的 POI 类别，例如具体店铺类型。
2. 稀有或格式不稳定的标签。
3. OSM 空白区域。

## 训练接入方式

P3A 中建议新增一个 `osm_weak_semantic` 辅助目标，输出 128x128 的多通道弱语义图：

- `building_density`
- `major_road_density`
- `minor_road_density`
- `water_density`
- `green_density`
- `residential_area`
- `commercial_area`
- `industrial_area`
- `agriculture_area`
- `poi_activity_density`

loss 用 BCE/Huber/Dice 的轻权重组合，并配 confidence mask。它只帮助 embedding 学到大概语义分布，不让 OSM 主导模型。
