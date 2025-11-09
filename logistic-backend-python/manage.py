import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import folium
from math import radians, sin, cos, sqrt, atan2

# Настройки
# Прямоугольная область вокруг Казани (bbox: запад, юг, восток, север)
bbox_kazan = (48.8, 55.6, 49.3, 55.9)

# Фильтр логистических объектов по типу здания
tags_logistics = {
    'building': ['warehouse', 'depot', 'industrial']
}

# Пути к локальным кэширующим файлам
osm_features_cache = "kazan_logistics_features.geojson"
osm_graph_cache = "kazan_drive.graphml"


# Загрузка или кэширование логистических объектов
if os.path.exists(osm_features_cache):
    centers_gdf = gpd.read_file(osm_features_cache)
    print("✅ Логистические объекты загружены из кэша")
else:
    print("🔍 Ищем логистические объекты в OpenStreetMap...")
    centers_gdf = ox.features.features_from_bbox(bbox=bbox_kazan, tags=tags_logistics)
    centers_gdf.to_file(osm_features_cache, driver="GeoJSON")
    print(f"💾 Найдено объектов: {len(centers_gdf)} и сохранено в кэш")

if centers_gdf.empty:
    print("❌ Не найдено логистических центров в заданной области.")
    exit()


# Получение координат объектов
coords = []
for _, row in centers_gdf.iterrows():
    geom = row.geometry
    # Для зданий используем центроид полигона
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        y, x = geom.centroid.y, geom.centroid.x
    else:
        y, x = geom.y, geom.x
    coords.append({
        'lat': y,
        'lon': x,
        'tags': row.to_dict()
    })

coords_df = pd.DataFrame(coords)
print(f"Получено {len(coords_df)} координат логистических точек")


# Функция геодезического расстояния
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радиус Земли в метрах
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# Формируем граф по прямым расстояниям
edges = []
for i, row_i in coords_df.iterrows():
    for j, row_j in coords_df.iterrows():
        if i < j:
            dist = haversine(row_i['lat'], row_i['lon'], row_j['lat'], row_j['lon'])
            edges.append((i, j, {'weight': dist}))

G = nx.Graph()
G.add_nodes_from(coords_df.index)
G.add_edges_from(edges)


# Минимальное остовное дерево
mst = nx.minimum_spanning_tree(G)
print(f" MST содержит {len(mst.nodes())} вершин и {len(mst.edges())} рёбер")


# Визуализация на карте
# Центр карты — середина bbox
m = folium.Map(
    location=[(bbox_kazan[1] + bbox_kazan[3]) / 2, (bbox_kazan[0] + bbox_kazan[2]) / 2],
    zoom_start=11
)

# Точки (логистические центры)
for i, row in coords_df.iterrows():
    tags = row['tags']
    name = tags.get('name', None)
    btype = tags.get('building', '—')
    street = tags.get('addr:street', '')
    housenumber = tags.get('addr:housenumber', '')
    city = tags.get('addr:city', '')

    # Формируем popup
    popup_lines = [f"<b>Тип:</b> {btype}"]
    if name:
        popup_lines.append(f"<b>Название:</b> {name}")
    if street or housenumber or city:
        address = ", ".join(filter(None, [city, street, housenumber]))
        popup_lines.append(f"<b>Адрес:</b> {address}")

    popup_html = "<br>".join(popup_lines)

    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        popup=folium.Popup(popup_html, max_width=500)
    ).add_to(m)

# Рёбра MST — прямые линии между складами
for u, v, data in mst.edges(data=True):
    row_u, row_v = coords_df.loc[u], coords_df.loc[v]
    folium.PolyLine(
        locations=[[row_u['lat'], row_u['lon']], [row_v['lat'], row_v['lon']]],
        color='blue',
        weight=2,
        opacity=0.6
    ).add_to(m)


output_file = "kazan_logistics_graph_mst.html"
m.save(output_file)
print(f"📄 Карта с MST маршрутов сохранена: {output_file}")
