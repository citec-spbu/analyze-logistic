import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import folium
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, List, Dict, Any


class LogisticsService:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.osm_features_cache = os.path.join(cache_dir, "kazan_logistics_features.geojson")
        self.osm_graph_cache = os.path.join(cache_dir, "kazan_drive.graphml")

        self.tags_logistics = {
            'building': ['warehouse', 'depot', 'industrial']
        }

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Вычисление геодезического расстояния между двумя точками"""
        R = 6371000  # Радиус Земли в метрах
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
        a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    def load_logistics_centers(self, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """Загрузка или кэширование логистических объектов"""
        if os.path.exists(self.osm_features_cache):
            centers_gdf = gpd.read_file(self.osm_features_cache)
            print("✅ Логистические объекты загружены из кэша")
        else:
            print("🔍 Ищем логистические объекты в OpenStreetMap...")
            centers_gdf = ox.features.features_from_bbox(bbox=bbox, tags=self.tags_logistics)
            centers_gdf.to_file(self.osm_features_cache, driver="GeoJSON")
            print(f"💾 Найдено объектов: {len(centers_gdf)} и сохранено в кэш")

        return centers_gdf

    def extract_coordinates(self, centers_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Извлечение координат из GeoDataFrame"""
        coords = []
        for _, row in centers_gdf.iterrows():
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

        return pd.DataFrame(coords)

    def build_graph(self, coords_df: pd.DataFrame) -> nx.Graph:
        """Построение графа по прямым расстояниям"""
        edges = []
        for i, row_i in coords_df.iterrows():
            for j, row_j in coords_df.iterrows():
                if i < j:
                    dist = self.haversine(
                        row_i['lat'], row_i['lon'],
                        row_j['lat'], row_j['lon']
                    )
                    edges.append((i, j, {'weight': dist}))

        G = nx.Graph()
        G.add_nodes_from(coords_df.index)
        G.add_edges_from(edges)
        return G

    def calculate_mst(self, G: nx.Graph) -> nx.Graph:
        """Вычисление минимального остовного дерева"""
        return nx.minimum_spanning_tree(G)

    def create_map(
            self,
            coords_df: pd.DataFrame,
            mst: nx.Graph,
            bbox: Tuple[float, float, float, float]
    ) -> folium.Map:
        """Создание интерактивной карты с MST"""
        # Центр карты
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        # Добавление точек
        for i, row in coords_df.iterrows():
            tags = row['tags']
            name = tags.get('name', None)
            btype = tags.get('building', '—')
            street = tags.get('addr:street', '')
            housenumber = tags.get('addr:housenumber', '')
            city = tags.get('addr:city', '')

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

        # Добавление рёбер MST
        for u, v, data in mst.edges(data=True):
            row_u, row_v = coords_df.loc[u], coords_df.loc[v]
            folium.PolyLine(
                locations=[[row_u['lat'], row_u['lon']], [row_v['lat'], row_v['lon']]],
                color='blue',
                weight=2,
                opacity=0.6
            ).add_to(m)

        return m

    def get_mst_data(self, coords_df: pd.DataFrame, mst: nx.Graph) -> Dict[str, Any]:
        """Получение данных MST для API ответа"""
        points = []
        for _, row in coords_df.iterrows():
            points.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'tags': row['tags']
            })

        edges = []
        total_distance = 0
        for u, v, data in mst.edges(data=True):
            distance = data['weight']
            total_distance += distance
            edges.append({
                'from_index': int(u),
                'to_index': int(v),
                'distance': float(distance)
            })

        return {
            'nodes_count': len(mst.nodes()),
            'edges_count': len(mst.edges()),
            'total_distance': float(total_distance),
            'points': points,
            'edges': edges
        }