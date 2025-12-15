import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import folium
from math import radians, sin, cos, sqrt, atan2, isnan
from typing import Tuple, Dict, Any, Optional
import pickle
import shutil
import atexit

from haversine import haversine, Unit
from scgraph.geographs.marnet import marnet_geograph

# =====================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================

def get_color(mode="auto"):
    if mode == "auto":
        return "#4B4B4B"
    elif mode == "rail":
        return "#6B3FA0"
    elif mode == "sea":
        return "#2F4F4F"
    elif mode == "aero":
        return "#708090"
    elif mode == "support":
        return "#8B6D5C"
    else:
        return "#7A7A7A"

def get_default_tags(mode: str) -> Dict[str, list]:
    """Возвращает набор OSM-тегов для логистических объектов по модам"""
    mode = mode.lower()
    if mode == "auto":
        return {"building": ["warehouse", "depot", "industrial"]}
    elif mode == "aero":
        return {"aeroway": ["terminal", "hangar", "cargo"]}
    elif mode == "sea":
        return {"harbour": True, "man_made": ["pier", "dock"]}
    elif mode == "rail":
        return {"railway": ["station", "yard", "cargo_terminal"]}
    else:
        raise ValueError(f"Неизвестный мод: {mode}")
    
def clear_cache_contents():
    if os.path.exists("cache"):
        for filename in os.listdir("cache"):
            file_path = os.path.join("cache", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Не удалось удалить {file_path}: {e}")
        print(f"Содержимое кэша  очищено")

atexit.register(clear_cache_contents)

# =====================
#  ОСНОВНЫЕ ФУНКЦИИ
# =====================

def load_logistics_features(
        bbox: Tuple[float, float, float, float],
        mode: str = "auto",
        cache_dir: str = "cache",
        cache_path: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Загружает или кэширует объекты логистической инфраструктуры"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "logistics.geojson")

    tags = get_default_tags(mode)

    # if os.path.exists(cache_path):
    #     print(f"Используем кэшированный файл: {cache_path}")
    #     gdf = gpd.read_file(cache_path)
    #     if not gdf.empty:
    #         return gdf
    #     print("Кэш пустой — перезагружаем.")

    print(f"Запрос в OSM ('{mode}')...")
    gdf = ox.features.features_from_bbox(bbox=bbox, tags=tags)
    gdf.to_file(cache_path, driver="GeoJSON")
    print(f"Сохранено объектов: {len(gdf)} → {cache_path}")
    return gdf

def clean_tags(row_dict):
    clean = {}
    for k, v in row_dict.items():
        if k == "geometry":
            continue
        if pd.isna(v):
            continue
        clean[k] = str(v)
    return clean

def extract_coordinates(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Извлекает координаты центроидов логистических объектов"""
    coords = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type in ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]:
            y, x = geom.centroid.y, geom.centroid.x
        else:
            y, x = geom.y, geom.x
        coords.append({
            "lat": float(y),
            "lon": float(x),
            "tags": clean_tags(row.to_dict())
        })
    return pd.DataFrame(coords)


def build_geodesic_graph(coords_df: pd.DataFrame) -> nx.Graph:
    """Создаёт граф, соединяя все точки прямыми (геодезическими) расстояниями."""
    edges = []
    for i, row_i in coords_df.iterrows():
        for j, row_j in coords_df.iterrows():
            if i < j:
                dist = haversine(
                    (row_i["lat"], row_i["lon"]),
                    (row_j["lat"], row_j["lon"]),
                    unit=Unit.KILOMETERS,  # или Unit.METERS
                )
                edges.append((i, j, {"weight": dist}))

    G = nx.Graph()
    G.add_nodes_from(coords_df.index)
    G.add_edges_from(edges)
    return G


def build_mst_graph(G: nx.Graph) -> nx.Graph:
    """Строит минимальное остовное дерево"""
    return nx.minimum_spanning_tree(G)


def build_mst_rail_by_color(coords_df: pd.DataFrame) -> nx.Graph:
    """
    Для mode='rail': строит отдельное MST для каждой линии метро (по colour)
    и отдельно MST для станций без цвета (NaN).
    """
    mst_total = nx.Graph()
    mst_total.add_nodes_from(coords_df.index)
    # группировка по цветам, включая nan-группу
    groups = coords_df.groupby(
        coords_df["tags"].apply(lambda t: t.get("colour") if "colour" in t else np.nan),
        dropna=False
    )

    for color, group_df in groups:
        if group_df.empty:
            continue

        color_label = color if pd.notna(color) else "NO_COLOR"
        print(f"Строим MST для линии '{color_label}' ({len(group_df)} станций)")

        # создаём локальную копию с переиндексацией (чтобы избежать «index out of range»)
        group_df_local = group_df.reset_index(drop=False)  # сохраним исходные индексы
        original_index = group_df_local["index"]

        # строим геодезический граф и MST
        subgraph = build_geodesic_graph(group_df_local)
        mst_color = nx.minimum_spanning_tree(subgraph)

        # переносим рёбра в общий MST с исходными индексами
        for u, v, data in mst_color.edges(data=True):
            idx_u = original_index.iloc[u]
            idx_v = original_index.iloc[v]
            mst_total.add_edge(
                idx_u,
                idx_v,
                weight=data["weight"],
                colour=None if pd.isna(color) else color
            )

    return mst_total

def build_sea_graph(coords_df: pd.DataFrame) -> nx.Graph:
    """
    Строит граф расстояний между морскими портами с учётом морских путей
    через библиотеку marnet_geograph.
    """
    G = nx.Graph()
    n = len(coords_df)
    G.add_nodes_from(coords_df.index)

    print("Используется marnet_geograph для расчёта морских расстояний...")

    for i in range(n):
        lat1, lon1 = coords_df.loc[i, ["lat", "lon"]]
        for j in range(i + 1, n):
            lat2, lon2 = coords_df.loc[j, ["lat", "lon"]]
            try:
                # Вызываем морской маршрут
                output = marnet_geograph.get_shortest_path(
                    origin_node={"latitude": lat1, "longitude": lon1},
                    destination_node={"latitude": lat2, "longitude": lon2}
                )
                # посчитаем длину траектории по координатам
                coord_path = output['coordinate_path']
                dist_total = 0.0
                for k in range(len(coord_path) - 1):
                    lat_a, lon_a = coord_path[k]
                    lat_b, lon_b = coord_path[k + 1]
                    dist_total += haversine((lat_a, lon_a), (lat_b, lon_b))

                G.add_edge(i, j, weight=dist_total)
            except Exception as e:
                print(f"Ошибка при расчёте пути ({i}-{j}): {e}")
                # fallback — просто геодезическое расстояние
                dist = haversine((lat1, lon1), (lat2, lon2))
                G.add_edge(i, j, weight=dist)
    return G

def create_base_map(bbox):
    m = folium.Map(
        location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
        zoom_start=12,
        zoom_control=False,
        tiles="OpenStreetMap",
        attr=""
    )

    m.get_root().html.add_child(folium.Element("""
        <style>
            .leaflet-control-attribution { display: none !important; }
        </style>
    """))

    return m

def draw_nodes_layer(m, coords_df):
    for _, row in coords_df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        tags = row.get("tags", {})
        name = tags.get("name")
        name_display = name if name and not pd.isna(name) else f"{row['lat']:.6f}, {row['lon']:.6f}"
        btype = tags.get("building", "—")

        popup = folium.Popup(
            f"<b>Название:</b> {name_display}<br><b>Тип:</b> {btype}",
            max_width=400
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="red",
            fill=True,
            fill_color="red",
            popup=popup
        ).add_to(m)

def draw_mst_layer(m, coords_df, mst, bbox, mode):
    """Рисует MST, безопасно обрабатывая ошибки авто/OSMnx"""
    fg = folium.FeatureGroup(name=f"MST: {mode}", show=True)

    G_drive = None
    # AUTO: пробуем OSMnx, но не падаем
    if mode == "auto" and len(coords_df) >= 2:
        try:
            G_drive = ox.graph_from_bbox(bbox, network_type="drive")
            if len(G_drive.nodes) == 0:
                G_drive = None
            else:
                coords_df = coords_df.copy()
                coords_df["osm_node"] = ox.distance.nearest_nodes(
                    G_drive, X=coords_df["lon"].values, Y=coords_df["lat"].values
                )
        except Exception as e:
            print(f"OSMnx авто режим недоступен: {e}")
            G_drive = None

    for u, v, data in mst.edges(data=True):
        ru, rv = coords_df.loc[u], coords_df.loc[v]

        # AUTO с OSMnx
        if mode == "auto" and G_drive is not None:
            try:
                route = ox.routing.shortest_path(G_drive, ru["osm_node"], rv["osm_node"], weight="length")
                route_gdf = ox.routing.route_to_gdf(G_drive, route)
                dist_km = route_gdf["length"].sum() / 1000
            except Exception:
                dist_km = haversine((ru["lat"], ru["lon"]), (rv["lat"], rv["lon"]))
            color = get_color(mode)
            weight = 3
        # RAIL
        elif mode == "all":
            dist_km = haversine((ru["lat"], ru["lon"]), (rv["lat"], rv["lon"]))
            color = data.get("colour") or "gray"
            weight = 3
            # делаем межмодовые линии заметнее

        elif mode == "rail":
            dist_km = haversine((ru["lat"], ru["lon"]), (rv["lat"], rv["lon"]))
            color = data.get("colour") or "blue"
            weight = 3
        # OTHER
        else:
            dist_km = haversine((ru["lat"], ru["lon"]), (rv["lat"], rv["lon"]))
            color, weight = "yellow", 3

        folium.PolyLine(
            [(ru["lat"], ru["lon"]), (rv["lat"], rv["lon"])],
            color=color,
            weight=weight,
            opacity=0.85,
            popup=f"{dist_km:.2f} км"
        ).add_to(fg)

    fg.add_to(m)


def visualize_mst_map(coords_df, mst, bbox, mode, output_file="logistics_mst.html"):
    m = create_base_map(bbox)
    draw_nodes_layer(m, coords_df)

    # --- Режим ALL: рисуем все моды слоями + межмодальные связи ---
    if mode == "all":
        # создадим отдельные слои по mode в coords_df
        modes_present = coords_df["mode"].unique() if "mode" in coords_df else []
        for mname in modes_present:
            sub_coords = coords_df[coords_df["mode"] == mname]
            fg = folium.FeatureGroup(name=f"{mname.upper()} network", show=True)
            # отрисуем рёбра только из этого мода
            for u, v, data in mst.edges(data=True):
                ru, rv = coords_df.loc[u], coords_df.loc[v]

                # Проверяем, оба ли ребра одного мода
                if ru["mode"] == rv["mode"] == mname:
                    color = get_color(mname)
                    dist_km = haversine((ru["lat"], ru["lon"]), (rv["lat"], rv["lon"]))
                    folium.PolyLine(
                        [(ru["lat"], ru["lon"]), (rv["lat"], rv["lon"])],
                        color=color,
                        weight=3,
                        opacity=0.9,
                        popup=f"{mname.upper()} {dist_km:.2f} км"
                    ).add_to(fg)
            fg.add_to(m)

        # --- отдельный слой для межмодальных рёбер ---
        intermodal_fg = folium.FeatureGroup(name="Межмодальные соединения", show=True)
        for u, v, data in mst.edges(data=True):
            ru, rv = coords_df.loc[u], coords_df.loc[v]
            if ru["mode"] != rv["mode"]:
                dist_km = haversine((ru["lat"], ru["lon"]), (rv["lat"], rv["lon"]))
                folium.PolyLine(
                    [(ru["lat"], ru["lon"]), (rv["lat"], rv["lon"])],
                    color="#ff8c00",
                    weight=3,
                    opacity=0.9,
                    popup=f"SEMI {dist_km:.2f} км"
                ).add_to(intermodal_fg)
        intermodal_fg.add_to(m)

    # --- остальные режимы без изменений ---
    else:
        draw_mst_layer(m, coords_df, mst, bbox, mode)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_file)
    print(f"Карта сохранена: {output_file}")
    return output_file

def generate_all_modes_mst(bbox, cache_dir="cache", output_file="logistics_mst_all.html"):
    """Безопасно строим MST для всех модов сразу"""
    modes = ["auto", "rail", "sea", "aero"]
    os.makedirs(cache_dir, exist_ok=True)

    m = create_base_map(bbox)
    all_coords = {}

    for mode in modes:
        res = generate_logistics_mst(bbox, mode, cache_dir)
        if res["status"] in ["ok", "no_data"]:
            try:
                coords_df = pd.read_pickle(os.path.join(cache_dir, f"coords_{mode}.pkl"))
                all_coords[mode] = coords_df
                mst_path = os.path.join(cache_dir, f"mst_{mode}.pkl")
                with open(mst_path, "rb") as f:
                    mst = pickle.load(f)
                draw_mst_layer(m, coords_df, mst, bbox, mode)
            except Exception as e:
                print(f"Ошибка MST для {mode}: {e}")

    # все точки
    combined_coords_df = pd.concat(all_coords.values(), ignore_index=True) if all_coords else pd.DataFrame()
    draw_nodes_layer(m, combined_coords_df)

    mst_all_path = os.path.join(cache_dir, "mst_all.pkl")
    if os.path.exists(mst_all_path):
        with open(mst_all_path, "rb") as f:
            G_all = pickle.load(f)
    else:
        G_all = nx.Graph()

    intermodal_fg = folium.FeatureGroup(name="Межмодальные соединения", show=True)
    for u, v, data in G_all.edges(data=True):
        if data.get("colour") == "gray":  # метка межмодальных связей
            ru, rv = coords_df.loc[u], coords_df.loc[v]
            folium.PolyLine(
                [(ru["lat"], ru["lon"]), (rv["lat"], rv["lon"])],
                color="#0000FF",     # ярко-синий
                weight=6,            # толще остальных
                opacity=0.9,
                popup=f"SEMI: {data.get('weight', 0):.2f} км"
            ).add_to(intermodal_fg)
    intermodal_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element("""
    <style>
        .leaflet-top.leaflet-right { top: 80px; }
    </style>
"""))
    m.save(output_file)
    print(f"Все MST сохранены на карте: {output_file}")
    return output_file

def compute_metric(G, metric):
    metric = metric.lower()

    if metric in ["degree", "degree_centrality"]:
        return nx.degree_centrality(G)

    elif metric in ["closeness", "closeness_centrality"]:
        return nx.closeness_centrality(G, distance="weight")

    elif metric in ["betweenness", "betweenness_centrality"]:
        return nx.betweenness_centrality(G, weight="weight", normalized=True)

    elif metric == "pagerank":
        return nx.pagerank(G, weight="weight", alpha=0.85)

    else:
        raise ValueError(f"Неизвестная метрика: {metric}")

def visualize_metric_map(coords_df, G, metric_vals, bbox, output_file="metric_map.html"):
    """Строит карту, где вершины окрашены согласно метрике"""

    m = folium.Map(
        location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
        zoom_start=11,
        zoom_control=False,
        tiles='OpenStreetMap',
        attr=''
    )

    m.get_root().html.add_child(folium.Element("""
        <style>
            .leaflet-top.leaflet-right {
                top: auto !important;   /* отменяем верхнее выравнивание */
                bottom: 20px !important; /* ставим снизу */
            }
        </style>
        """))

    # Нормализация значений для цветов
    values = np.array(list(metric_vals.values()))
    vmin, vmax = values.min(), values.max()

    def color_for_value(v):
        """Градиент зелёный -> красный"""
        if vmax == vmin:
            t = 0
        else:
            t = (v - vmin) / (vmax - vmin)

        r = int(255 * t)
        g = int(255 * (1 - t))
        return f"#{r:02x}{g:02x}00"

    # Рёбра графа 
    for u, v, _ in G.edges(data=True):
        ru = coords_df.loc[u]
        rv = coords_df.loc[v]

        folium.PolyLine(
            [(ru["lat"], ru["lon"]), (rv["lat"], rv["lon"])],
            color="#1F1E1E",
            weight=1,
            opacity=1
        ).add_to(m)

    # Вершины
    for idx, row in coords_df.iterrows():
        if idx not in metric_vals:
            continue

        value = metric_vals[idx]
        color = color_for_value(value)

        tags = row["tags"]
        name = tags.get("name")
        btype = tags.get("building", "—")

        popup = folium.Popup(
            f"<b>Метрика:</b> {value:.4f}<br>"
            f"<b>Тип:</b> {btype}<br>"
            f"<b>Название:</b> {name}",
            max_width=400
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup
        ).add_to(m)

    m.save(output_file)
    return output_file



# =====================
#  ГЛАВНАЯ ФУНКЦИЯ API
# =====================

def generate_logistics_mst(bbox, mode="auto", cache_dir="cache", output_file=None):
    """Генерация MST для одного режима с полной защитой от ошибок Overpass/OSMnx"""
    os.makedirs(cache_dir, exist_ok=True)

    if mode == "all":
        print("Режим ALL: строим MST для каждого мода и связываем с авто-сетью")

        modes = ["auto", "rail", "sea", "aero"]
        all_coords, all_msts = {}, {}

        # генерируем MST по всем доступным модам
        for m in modes:
            res = generate_logistics_mst(bbox, m, cache_dir)
            if res["status"] == "ok":
                coords_df = pd.read_pickle(os.path.join(cache_dir, f"coords_{m}.pkl"))
                all_coords[m] = coords_df
                with open(os.path.join(cache_dir, f"mst_{m}.pkl"), "rb") as f:
                    all_msts[m] = pickle.load(f)

        if "auto" not in all_coords or all_coords["auto"].empty:
            return {"status": "error", "message": "Нет данных для соединения с модом 'auto'."}

        # итоговый граф: объединяем все MST
        G_all = nx.Graph()
        node_offset = 0
        index_map = {}
        combined_coords = pd.DataFrame(columns=["lat", "lon", "tags", "mode"])

        # объединяем координаты и MST всех модов в один граф
        for m, coords_df in all_coords.items():
            coords_df = coords_df.copy()
            coords_df["mode"] = m

            # вычисляем глобальные индексы правильно
            start_idx = len(combined_coords)
            idx_map = {old: start_idx + i for i, old in enumerate(coords_df.index)}
            index_map[m] = idx_map

            coords_df.index = [start_idx + i for i in range(len(coords_df))]  # глобальные индексы
            combined_coords = pd.concat([combined_coords, coords_df])

            mst = all_msts.get(m)
            if mst:
                for u, v, data in mst.edges(data=True):
                    G_all.add_edge(idx_map[u], idx_map[v], **data)

        # Добавляем связи "aero/sea/rail" <-> "auto"
        print("Соединяем узлы aero/rail/sea с ближайшими auto узлами...")
        auto_df = combined_coords[combined_coords["mode"] == "auto"].copy()

        # ✳️ фикс 1 — сбрасываем индекс и сохраняем соответствие с combined_coords
        auto_df = auto_df.reset_index().rename(columns={"index": "global_index"})

        if len(auto_df) >= 2:
            try:
                G_drive = ox.graph_from_bbox(bbox, network_type="drive")
                auto_df["osm_node"] = ox.distance.nearest_nodes(
                    G_drive, X=auto_df["lon"].values, Y=auto_df["lat"].values
                )
            except Exception as e:
                print(f"Ошибка загрузки дорожного графа: {e}")
                G_drive = None
        else:
            G_drive = None

        for m in ["aero", "rail", "sea"]:
            if m not in all_coords or all_coords[m].empty:
                continue
            df = all_coords[m]

            print(f"🔗 Соединяем все узлы '{m}' с ближайшими 'auto' по автодорогам...")

            for idx, row in df.iterrows():
                lat, lon = row["lat"], row["lon"]

                # ищем ближайший авто-пункт
                auto_dists = auto_df.apply(lambda r: haversine((lat, lon), (r["lat"], r["lon"])), axis=1)
                nearest_idx = auto_dists.idxmin()
                nearest_global_auto_idx = int(auto_df.loc[nearest_idx, "global_index"])
                lat2, lon2 = auto_df.loc[nearest_idx, ["lat", "lon"]]
                dist_km = haversine((lat, lon), (lat2, lon2))

                # если возможно, уточняем длину по дороге
                if G_drive is not None:
                    try:
                        from_node = ox.distance.nearest_nodes(G_drive, lon, lat)
                        to_node = auto_df.loc[nearest_idx, "osm_node"]
                        route = ox.routing.shortest_path(G_drive, from_node, to_node, weight="length")
                        if route and len(route) > 1:
                            route_gdf = ox.routing.route_to_gdf(G_drive, route)
                            dist_km = route_gdf["length"].sum() / 1000
                    except Exception as e:
                        print(f"Предупреждение: не удалось рассчитать путь для {m}→auto: {e}")

                # добавляем ребро в объединённый граф
                idx_all_m = index_map[m][idx]
                G_all.add_edge(
                    idx_all_m,
                    nearest_global_auto_idx,
                    weight=dist_km,
                    colour="gray"
                )

        # сохраняем результаты
        coords_path = os.path.join(cache_dir, "coords_all.pkl")
        mst_path = os.path.join(cache_dir, "mst_all.pkl")
        mst_map_path = output_file or os.path.join(cache_dir, "mst_all.html").replace("\\", "/")

        combined_coords.to_pickle(coords_path)
        with open(mst_path, "wb") as f:
            pickle.dump(G_all, f)

        highlight_edges = [(u, v, d) for u, v, d in G_all.edges(data=True) if d.get("colour") == "gray"]
        if highlight_edges:
            print(f"Добавлено {len(highlight_edges)} межмодальных соединений.")
        else:
            print("⚠ Межмодальные соединения не найдены!")

        visualize_mst_map(combined_coords, G_all, bbox, "all", output_file=mst_map_path)

        edges = [
            {"from_index": int(u), "to_index": int(v), "distance": float(data.get("weight", 0))}
            for u, v, data in G_all.edges(data=True)
        ]
        points = [
            {"lat": float(r["lat"]), "lon": float(r["lon"]), "tags": r["tags"], "mode": r["mode"]}
            for _, r in combined_coords.iterrows()
        ]
        total_distance = sum(e["distance"] for e in edges)

        return {
            "status": "ok",
            "mode": "all",
            "bbox": bbox,
            "map_path": mst_map_path,
            "coords_path": coords_path,
            "mst_path": mst_path,
            "points": points,
            "edges": edges,
            "total_distance": total_distance,
            "nodes_count": len(points),
            "edges_count": len(edges)
        }

    coords_path = os.path.join(cache_dir, f"coords_{mode}.pkl")
    mst_path = os.path.join(cache_dir, f"mst_{mode}.pkl")
    mst_map_path = output_file or os.path.join(cache_dir, f"mst_{mode}.html").replace("\\", "/")

    try:
        gdf = None
        try:
            gdf = load_logistics_features(bbox, mode, cache_dir)
        except Exception as e:
            print(f"Ошибка загрузки OSM для {mode}: {e}")
            gdf = gpd.GeoDataFrame(columns=["geometry", "tags"])

        if gdf.empty or len(gdf) < 2:
            print(f"Недостаточно объектов для {mode}, строим MST с фиктивными точками (0 точек → пустой MST)")
            coords_df = pd.DataFrame(columns=["lat", "lon", "tags"])
            mst = nx.Graph()
        else:
            coords_df = extract_coordinates(gdf)
            if mode == "rail":
                mst = build_mst_rail_by_color(coords_df)
            else:
                G = build_geodesic_graph(coords_df)
                mst = build_mst_graph(G)

        coords_df.to_pickle(coords_path)
        with open(mst_path, "wb") as f: pickle.dump(mst, f)

        # карта
        try:
            visualize_mst_map(coords_df, mst, bbox, mode, output_file=mst_map_path)
        except Exception as e:
            print(f"Ошибка визуализации MST: {e}")

        points = [
            {"lat": float(row["lat"]), "lon": float(row["lon"]),
             "tags": {k: (None if pd.isna(v) else str(v)) for k, v in row["tags"].items() if k != "geometry"}}
            for _, row in coords_df.iterrows()
        ]

        edges = [
            {"from_index": int(u), "to_index": int(v), "distance": float(data["weight"])}
            for u, v, data in mst.edges(data=True)
        ]

        total_distance = sum(edge["distance"] for edge in edges)

        return {
            "status": "ok",
            "map_path": mst_map_path,
            "coords_path": coords_path,
            "mst_path": mst_path,
            "bbox": bbox,
            "mode": mode,
            "points": points,
            "edges": edges,
            "total_distance": total_distance,
            "nodes_count": len(points),
            "edges_count": len(edges)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def analyze_logistics_metrics(bbox, mode, metric, cache_dir="cache"):
    """
    Анализ метрик для одного или всех модов. Для 'all' используется объединённый граф.
    """

    # --- Поддержка режима ALL ---
    if mode == "all":
        coords_path = os.path.join(cache_dir, "coords_all.pkl")
        mst_path = os.path.join(cache_dir, "mst_all.pkl")
    else:
        coords_path = os.path.join(cache_dir, f"coords_{mode}.pkl")
        mst_path = os.path.join(cache_dir, f"mst_{mode}.pkl")

    metric_map_path = os.path.join(cache_dir, f"metric_{mode}_{metric}.html").replace("\\", "/")

    if not os.path.exists(coords_path) or not os.path.exists(mst_path):
        return {"status": "error", "message": f"MST не найден для режима {mode}."}

    coords_df = pd.read_pickle(coords_path)
    with open(mst_path, "rb") as f:
        G = pickle.load(f)

    if G.number_of_nodes() == 0:
        return {"status": "error", "message": f"MST пустой для режима {mode}."}

    if G.number_of_nodes() < 2:
        return {"status": "error", "message": "Недостаточно вершин для расчёта метрик."}

    # --- Вычисляем метрику ---
    try:
        metric_vals = compute_metric(G, metric)
    except Exception as e:
        return {"status": "error", "message": f"Ошибка вычисления метрики: {e}"}

    # --- Визуализация ---
    # Для 'all' цвета брать из mode узла, межмодальные ребра окрашивать отдельно
    if mode == "all":
        m = create_base_map(bbox)

        # добавить рёбра (все)
        for u, v, data in G.edges(data=True):
            ru, rv = coords_df.loc[u], coords_df.loc[v]
            weight = 3
            color = get_color(ru["mode"])
            folium.PolyLine(
                [(ru["lat"], ru["lon"]), (rv["lat"], rv["lon"])],
                color=color,
                weight=weight,
                opacity=0.7
            ).add_to(m)

        # нормализуем метрики
        values = np.array(list(metric_vals.values()))
        vmin, vmax = values.min(), values.max()

        def color_for_value(v):
            if vmax == vmin:
                t = 0
            else:
                t = (v - vmin) / (vmax - vmin)
            r = int(255 * t)
            g = int(255 * (1 - t))
            return f"#{r:02x}{g:02x}00"

        # добавить вершины
        for idx, row in coords_df.iterrows():
            if idx not in metric_vals:
                continue
            val = metric_vals[idx]
            color = color_for_value(val)
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=f"{row['mode'].upper()} | {metric}: {val:.4f}"
            ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(metric_map_path)

    else:
        # старый случай для одиночных модов
        visualize_metric_map(coords_df, G, metric_vals, bbox, output_file=metric_map_path)

    metric_vals_clean = {int(k): float(v) for k, v in metric_vals.items()}

    return {
        "status": "ok",
        "metric": metric,
        "map_path": metric_map_path,
        "bbox": bbox,
        "mode": mode,
        "nodes_count": len(coords_df),
        "values": metric_vals_clean
    }