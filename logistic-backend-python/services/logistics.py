import os
from typing import Tuple, Dict, Any, Optional

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from haversine import haversine, Unit


# =====================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================


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

    # точки
    for i, row in coords_df.iterrows():
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

    # рёбра и подписи расстояний
    for u, v, data in mst.edges(data=True):
        row_u, row_v = coords_df.loc[u], coords_df.loc[v]
        dist_m = float(data["weight"])
        dist_km = dist_m / 1000.0

        folium.PolyLine(
        locations=[[row_u["lat"], row_u["lon"]], [row_v["lat"], row_v["lon"]]],
        color="blue",
        weight=2,
        opacity=0.6,
        popup=f"Расстояние: {dist_km:.2f} км"
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
        mode: str,
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
    html_path = visualize_mst_map(coords_df, mst, bbox, mode, output_file)

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
