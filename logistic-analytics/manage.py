import osmnx as ox
import folium
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import networkx as nx

# -------------------------
# Настройки
# -------------------------
bbox_kazan = (48.8, 55.6, 49.3, 55.9)

tags_logistics = {
    'building': ['warehouse', 'depot', 'industrial']
}

# -------------------------
# 1) Получаем логистические объекты
# -------------------------
print("Ищем логистические объекты...")
centers_gdf = ox.features.features_from_bbox(bbox=bbox_kazan, tags=tags_logistics)
print(f"Найдено объектов: {len(centers_gdf)}")

if centers_gdf.empty:
    print("Не найдено логистических центров.")
    exit()

# -------------------------
# 2) Получаем координаты и информацию по каждому объекту
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
# 3) Кластеризация точек (DBSCAN)
# -------------------------
kms_per_radian = 6371.0088
epsilon = 0.15 / kms_per_radian  # 150 метров

coords_rad = np.radians(coords_df[['lat','lon']])
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
coords_df['cluster'] = db.labels_

# -------------------------
# 4) Агрегируем центроиды и информацию по кластерам
# -------------------------
clustered_centers = coords_df.groupby('cluster').agg({
    'lat': 'mean',
    'lon': 'mean',
    'tags': lambda x: list(x)
}).reset_index()

clustered_centers['name'] = ['Логистический центр ' + str(i+1) for i in clustered_centers.index]
print(f"После кластеризации центров: {len(clustered_centers)}")

# -------------------------
# 5) Загружаем дорожную сеть (drive)
# -------------------------
print("Загружаем дорожную сеть...")
G_drive = ox.graph_from_bbox(
    bbox=bbox_kazan,
    network_type='drive'
)

# -------------------------
# 6) Строим граф расстояний между кластерами
# -------------------------
edges = []
nodes = clustered_centers.index.tolist()

for i in nodes:
    for j in nodes:
        if i < j:
            try:
                node_i = ox.distance.nearest_nodes(G_drive, X=clustered_centers.loc[i,'lon'], Y=clustered_centers.loc[i,'lat'])
                node_j = ox.distance.nearest_nodes(G_drive, X=clustered_centers.loc[j,'lon'], Y=clustered_centers.loc[j,'lat'])
                length = nx.shortest_path_length(G_drive, node_i, node_j, weight='length')
                edges.append((i, j, {'weight': length}))
            except:
                # если путь не найден, пропускаем
                continue

# создаём граф NetworkX
G_clusters = nx.Graph()
G_clusters.add_nodes_from(nodes)
G_clusters.add_edges_from(edges)

# -------------------------
# 7) Строим минимальный остов (MST) для упрощённого графа
# -------------------------
mst = nx.minimum_spanning_tree(G_clusters)

# -------------------------
# 8) Визуализация на Folium
# -------------------------
m = folium.Map(
    location=[(bbox_kazan[1]+bbox_kazan[3])/2, (bbox_kazan[0]+bbox_kazan[2])/2],
    zoom_start=11
)

# добавляем вершины
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

# добавляем рёбра MST
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
