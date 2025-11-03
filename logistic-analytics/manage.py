import osmnx as ox
import folium
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

# -------------------------
# Настройки
# -------------------------
# bbox Казани (west, south, east, north)
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

# Получаем координаты центров (центроид для полигона)
coords = []
for geom in centers_gdf.geometry:
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        coords.append([geom.centroid.y, geom.centroid.x])
    else:
        coords.append([geom.y, geom.x])

coords = pd.DataFrame(coords, columns=['lat', 'lon'])

# -------------------------
# 2) Кластеризация точек (DBSCAN)
# -------------------------
kms_per_radian = 6371.0088
epsilon = 0.15 / kms_per_radian  # 150 метров

coords_rad = np.radians(coords[['lat','lon']])
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
coords['cluster'] = db.labels_

# агрегируем центроиды кластеров
clustered_centers = coords.groupby('cluster').mean().reset_index()
clustered_centers['name'] = ['Логистический центр ' + str(i+1) for i in clustered_centers.index]
print(f"После кластеризации центров: {len(clustered_centers)}")

# -------------------------
# 3) Загружаем дорожную сеть
# -------------------------
print("Загружаем дорожную сеть (drive)...")
G_drive = ox.graph_from_bbox(
    bbox=bbox_kazan,
    network_type='drive'
)
# длины рёбер ('length') уже есть в графе

# -------------------------
# 4) Находим ближайшие узлы к центрам
# -------------------------
clustered_centers['node'] = clustered_centers.apply(
    lambda row: ox.distance.nearest_nodes(G_drive, X=row['lon'], Y=row['lat']), axis=1
)

# -------------------------
# 5) Строим кратчайшие маршруты между центрами
# -------------------------
routes = []
for i, row1 in clustered_centers.iterrows():
    for j, row2 in clustered_centers.iterrows():
        if i < j:
            route = ox.shortest_path(G_drive, row1['node'], row2['node'], weight='length')
            routes.append((i, j, route))

# -------------------------
# 6) Визуализация на Folium
# -------------------------
m = folium.Map(
    location=[(bbox_kazan[1]+bbox_kazan[3])/2, (bbox_kazan[0]+bbox_kazan[2])/2],
    zoom_start=11
)

# добавляем центры
for idx, row in clustered_centers.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=row['name']
    ).add_to(m)

# добавляем маршруты
for i, j, route in routes:
    coords_route = [(G_drive.nodes[n]['y'], G_drive.nodes[n]['x']) for n in route]
    folium.PolyLine(coords_route, color='blue', weight=2, opacity=0.5).add_to(m)

m.save("kazan_logistics_graph.html")
print("Карта с графом сохранена: kazan_logistics_graph.html")
