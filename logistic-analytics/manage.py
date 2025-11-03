import osmnx as ox
import folium

# bbox Казани (west, south, east, north)
bbox_kazan = (48.8, 55.6, 49.3, 55.9)

tags_logistics = {
    'building': ['warehouse', 'depot', 'industrial'],
    #'amenity': ['fuel', 'parking', 'truck_stop', 'marketplace', 'post_office']
}

print("Ищем логистические объекты...")
centers_gdf = ox.features.features_from_bbox(bbox=bbox_kazan, tags=tags_logistics)
print(f"Найдено объектов: {len(centers_gdf)}")

# Если есть объекты, визуализируем через Folium
if not centers_gdf.empty:
    m = folium.Map(location=[(bbox_kazan[1]+bbox_kazan[3])/2, (bbox_kazan[0]+bbox_kazan[2])/2], zoom_start=11)
    for idx, row in centers_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            geom = geom.centroid
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=str(row.get('building', row.get('amenity', 'неизвестно')))
        ).add_to(m)
    m.save("kazan_logistics_map.html")
    print("Карта сохранена: kazan_logistics_map.html")
else:
    print("Не найдено логистических центров.")
