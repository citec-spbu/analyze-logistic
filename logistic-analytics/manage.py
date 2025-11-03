import osmnx as ox
import folium
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import os
import geopandas as gpd

# -------------------------
# Настройки
# -------------------------
bbox_kazan = (48.8, 55.6, 49.3, 55.9)
tags_logistics = {
    'building': ['warehouse', 'depot', 'industrial']
}

# Пути к кэшу
osm_features_cache = "kazan_logistics_features.geojson"
osm_graph_cache = "kazan_drive.graphml"

# -------------------------
# 1) Загружаем логистические объекты
# -------------------------
if os.path.exists(osm_features_cache):
    centers_gdf = gpd.read_file(osm_features_cache)
    print("Объекты загружены из кэша")
else:
    print("Ищем логистические объекты...")
    centers_gdf = ox.features.features_from_bbox(bbox=bbox_kazan, tags=tags_logistics)
    centers_gdf.to_file(osm_features_cache, driver="GeoJSON")
    print(f"Найдено объектов: {len(centers_gdf)} и сохранено в кэш")

if centers_gdf.empty:
    print("Не найдено логистических центров.")
    exit()

# -------------------------
# 2) Получаем координаты и информацию
# -------------------------
coords = []
for idx, row in centers_gdf.iterrows():
    geom = row.geometry
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

# -------------------------
# 3) Кластеризация DBSCAN
# -------------------------
kms_per_radian = 6371.0088
epsilon = 0.15 / kms_per_radian  # ~150 м
coords_rad = np.radians(coords_df[['lat','lon']])
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
coords_df['cluster'] = db.labels_

# -------------------------
# 4) Агрегируем кластеры
# -------------------------
clustered_centers = coords_df.groupby('cluster').agg({
    'lat': 'mean',
    'lon': 'mean',
    'tags': lambda x: list(x)
}).reset_index()
clustered_centers['name'] = ['Логистический центр ' + str(i+1) for i in clustered_centers.index]
print(f"После кластеризации центров: {len(clustered_centers)}")

# -------------------------
# 5) Загружаем граф дорог
# -------------------------
if os.path.exists(osm_graph_cache):
    G_drive = ox.load_graphml(osm_graph_cache)
    print("Граф загружен из кэша")
else:
    print("Загружаем дорожную сеть...")
    G_drive = ox.graph_from_bbox(bbox=bbox_kazan, network_type='drive')
    ox.save_graphml(G_drive, osm_graph_cache)
    print("Граф скачан и сохранён в кэш")

# -------------------------
# 6) Находим ближайшие узлы для кластеров
# -------------------------
clustered_centers['nearest_node'] = clustered_centers.apply(
    lambda row: ox.distance.nearest_nodes(G_drive, X=row['lon'], Y=row['lat']), axis=1
)

# -------------------------
# 7) Строим граф кластеров с весами (только между центрами)
# -------------------------
nodes_list = clustered_centers['nearest_node'].tolist()
edges = []

# Вычисляем кратчайшие пути только между кластерами
for i, row_i in clustered_centers.iterrows():
    lengths = nx.single_source_dijkstra_path_length(G_drive, row_i['nearest_node'], weight='length')
    for j, row_j in clustered_centers.iterrows():
        if i < j:
            length = lengths.get(row_j['nearest_node'], np.inf)
            if np.isfinite(length):
                edges.append((i, j, {'weight': length}))

# Создаём граф NetworkX
G_clusters = nx.Graph()
G_clusters.add_nodes_from(clustered_centers.index.tolist())
G_clusters.add_edges_from(edges)

# -------------------------
# 8) Минимальный остов (MST)
# -------------------------
mst = nx.minimum_spanning_tree(G_clusters)

# -------------------------
# 9) Визуализация на Folium
# -------------------------
m = folium.Map(
    location=[(bbox_kazan[1]+bbox_kazan[3])/2, (bbox_kazan[0]+bbox_kazan[2])/2],
    zoom_start=11
)

# Вершины с информацией
for idx, row in clustered_centers.iterrows():
    popup_html = f"<b>{row['name']}</b><br>"
    for bld in row['tags']:
        building_type = bld.get('building', 'неизвестно')
        name = bld.get('name', '')
        popup_html += f"{building_type} {name}<br>"

    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=popup_html
    ).add_to(m)

# MST рёбра
for u, v, data in mst.edges(data=True):
    row_u = clustered_centers.loc[u]
    row_v = clustered_centers.loc[v]
    folium.PolyLine(
        locations=[[row_u['lat'], row_u['lon']], [row_v['lat'], row_v['lon']]],
        color='blue',
        weight=2,
        opacity=0.6
    ).add_to(m)

m.save("kazan_logistics_graph_mst.html")
print("Карта с MST маршрутов сохранена: kazan_logistics_graph_mst.html")
