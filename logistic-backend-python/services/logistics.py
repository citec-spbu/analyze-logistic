import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import folium
from math import radians, sin, cos, sqrt, atan2, isnan
from typing import Tuple, Dict, Any, Optional

from haversine import haversine, Unit


# =====================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================

# def haversine(lat1, lon1, lat2, lon2):
#     """Геодезическое расстояние между точками в метрах"""
#     R = 6371000
#     phi1, phi2 = radians(lat1), radians(lat2)
#     dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
#     a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
#     return 2 * R * atan2(sqrt(a), sqrt(1 - a))


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


# =====================
#  ОСНОВНЫЕ ФУНКЦИИ
# =====================

def load_logistics_features(
        bbox: Tuple[float, float, float, float],
        mode: str = "auto",
        cache_path: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Загружает или кэширует объекты логистической инфраструктуры"""
    tags = get_default_tags(mode)
    cache_path = cache_path or f"logistics_{mode}_features.geojson"

    if False:
        gdf = gpd.read_file(cache_path)
        print(f"Загружено из кэша: {cache_path}")
    else:
        print(f"Запрос к OSM для режима '{mode}'...")
        gdf = ox.features.features_from_bbox(bbox=bbox, tags=tags)
        gdf.to_file(cache_path, driver="GeoJSON")
        print(f"Найдено объектов: {len(gdf)} (сохранено в {cache_path})")

    return gdf


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
            "lat": y,
            "lon": x,
            "tags": row.to_dict()
        })
    return pd.DataFrame(coords)


def build_geodesic_graph(coords_df: pd.DataFrame) -> nx.Graph:
    """Создаёт граф, соединяя все точки прямыми (геодезическими) расстояниями."""
    edges = []
    for i, row_i in coords_df.iterrows():
        for j, row_j in coords_df.iterrows():
            if i < j:
                # ✅ обязательно передаём кортежи (lat, lon)
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

    # → группировка по цветам, включая nan-группу
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

def visualize_mst_map(coords_df, mst, bbox, mode, output_file="logistics_mst.html"):
    """
    Отображает MST на карте Folium.
    Для mode='rail' линии имеют цвета по атрибуту 'colour',
    для mode='auto' и других — просто зелёные/синие.
    """
    # Центр карты
    m = folium.Map(
        location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
        zoom_start=12
    )

    # --- точки ---
    for _, row in coords_df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        tags = row.get("tags", {})
        name = tags.get("name")
        btype = tags.get("building", "—")

        popup_lines = [f"<b>Тип:</b> {btype}"]
        if name and not pd.isna(name):
            popup_lines.append(f"<b>Название:</b> {name}")

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=6, color="red", fill=True, fill_color="red",
            popup=folium.Popup("<br>".join(popup_lines), max_width=500)
        ).add_to(m)

    print(f"📥 Отрисовка рёбер для mode='{mode}' ...")

    # --- рёбра ---
    for u, v, data in mst.edges(data=True):
        row_u = coords_df.loc[u]
        row_v = coords_df.loc[v]

        dist_hav = haversine(
            (row_u["lat"], row_u["lon"]),
            (row_v["lat"], row_v["lon"])
        )
        popup_html = f"<b>Прямое расстояние:</b> {dist_hav:.2f}&nbsp;км"

        edge_color = data.get("colour")
        if pd.isna(edge_color) or not edge_color:
            edge_color = "gray"
        else:
            edge_color = str(edge_color).strip().lower()

        folium.PolyLine(
            locations=[[row_u["lat"], row_u["lon"]], [row_v["lat"], row_v["lon"]]],
            color=edge_color,
            weight=4,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    # сохраняем на диск
    m.save(output_file)
    print(f"📄 Карта сохранена: {output_file}")
    return output_file

# =====================
#  ГЛАВНАЯ ФУНКЦИЯ API
# =====================

def generate_logistics_mst(
        bbox: Tuple[float, float, float, float],
        mode: str = "auto",
        cache_dir: str = ".",
        output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Главная функция: строит MST и возвращает полные данные"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"logistics_{mode}_features.geojson")
    output_file = output_file or os.path.join(cache_dir, f"logistics_{mode}_mst.html")

    gdf = load_logistics_features(bbox, mode, cache_path)
    if gdf.empty:
        return {"status": "no_data", "message": "Нет логистических объектов в области."}

    coords_df = extract_coordinates(gdf)
    # G = build_geodesic_graph(coords_df)
    # mst = build_mst_graph(G)
    if mode == "rail":
        mst = build_mst_rail_by_color(coords_df)
    else:
        G = build_geodesic_graph(coords_df)
        mst = build_mst_graph(G)
    html_path = visualize_mst_map(coords_df, mst, bbox, output_file)

    # Формируем полную структуру MST
    points = []
    for _, row in coords_df.iterrows():
        clean_tags = {}
        for k, v in row["tags"].items():
            if k == "geometry":
                continue
            if pd.isna(v):
                clean_tags[k] = None
            else:
                clean_tags[k] = str(v)

        points.append({
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "tags": clean_tags
        })

    edges = []
    total_distance = 0.0
    for u, v, data in mst.edges(data=True):
        d = float(data["weight"])
        total_distance += d
        edges.append({
            "from_index": int(u),
            "to_index": int(v),
            "distance": d
        })

    return {
        "nodes_count": len(points),
        "edges_count": len(edges),
        "total_distance": total_distance,
        "points": points,
        "edges": edges,
        "map_path": html_path,
        "mode": mode,
        "bbox": bbox,
        "status": "ok"
    }

