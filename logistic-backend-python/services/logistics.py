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
        print(f"✅ Загружено из кэша: {cache_path}")
    else:
        print(f"🔍 Запрос к OSM для режима '{mode}'...")
        gdf = ox.features.features_from_bbox(bbox=bbox, tags=tags)
        gdf.to_file(cache_path, driver="GeoJSON")
        print(f"💾 Найдено объектов: {len(gdf)} (сохранено в {cache_path})")

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


def visualize_mst_map(coords_df, mst, bbox, mode, output_file="logistics_mst.html"):
    """
    Отображает MST на карте Folium.
    Для mode='auto' — длина по дорогам,
    для других mode — длина прямой между точками.
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

    print(f"📥 Загрузка дорожной сети для mode='{mode}' ...")
    G_drive = ox.graph_from_bbox(bbox, network_type="drive")
    print(f"✅ Граф: узлов={len(G_drive.nodes)}, рёбер={len(G_drive.edges)}")

    coords_df = coords_df.copy()
    coords_df["osm_node"] = ox.distance.nearest_nodes(
        G_drive,
        X=coords_df["lon"].values,
        Y=coords_df["lat"].values
    )

    print("🚗 Построение маршрутов ...")
    for u, v, _ in mst.edges(data=True):
        row_u, row_v = coords_df.loc[u], coords_df.loc[v]

        # если mode != 'auto', то считаем только по прямой
        if mode != "auto":
            dist_hav = haversine(
                (row_u["lat"], row_u["lon"]),
                (row_v["lat"], row_v["lon"])
            )

            popup_html = f"<b>Прямое расстояние:</b> {dist_hav:.2f}&nbsp;км"
            folium.PolyLine(
                locations=[[row_u["lat"], row_u["lon"]], [row_v["lat"], row_v["lon"]]],
                color="green", weight=3, opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
            continue

        # иначе (mode == 'auto') считаем по дорогам
        node_u = row_u["osm_node"]
        node_v = row_v["osm_node"]
        try:
            route = ox.routing.shortest_path(G_drive, node_u, node_v, weight="length", cpus=4)
        except Exception:
            route = None

        if route and len(route) > 1:
            route_gdf = ox.routing.route_to_gdf(G_drive, route)
            dist_m = float(route_gdf["length"].sum())
            dist_km = dist_m / 1000.0
            popup_html = f"<b>Расстояние по дорогам:</b> {dist_km:.2f}&nbsp;км"
            color = "blue"
        else:
            dist_hav = haversine(
                (row_u["lat"], row_u["lon"]),
                (row_v["lat"], row_v["lon"])
            ) / 1000.0
            popup_html = f"<b>Прямое расстояние:</b> {dist_hav:.2f}&nbsp;км"
            color = "gray"

        folium.PolyLine(
            locations=[[row_u["lat"], row_u["lon"]], [row_v["lat"], row_v["lon"]]],
            color=color, weight=3, opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    m.save(output_file)
    print(f"📄 Карта сохранена: {output_file}")
    return output_file


# =====================
#  ГЛАВНАЯ ФУНКЦИЯ API
# =====================

import pandas as pd  # добавь импорт наверху, если его нет


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

