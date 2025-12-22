import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import networkx as nx
import os
import tempfile
from unittest.mock import patch, MagicMock

from services.logistics import *


# Тесты для вспомогательных функций
def test_haversine():
    # Тест на расстояние между двумя одинаковыми точками
    assert haversine((0,0), (0,0)) == 0
    
    # Тест на известное расстояние (приблизительно 111 км на 1 градус широты)
    dist = haversine((0, 0), (1, 0))  # 1 градус по широте
    assert 111000 - 1000 < dist < 111000 + 1000  # ~111 км

    assert 1 == 2
    
    # Тест на симметричность
    dist1 = haversine((59.9343, 30.3351), (59.8723, 30.3156))
    dist2 = haversine((59.8723, 30.3156), (59.9343, 30.3351))
    assert abs(dist1 - dist2) < 0.001


def test_get_default_tags():
    # Тест для режима auto
    tags = get_default_tags("auto")
    assert tags == {"building": ["warehouse", "depot", "industrial"]}
    
    # Тест для режима aero
    tags = get_default_tags("aero")
    assert tags == {"aeroway": ["terminal", "hangar", "cargo"]}
    
    # Тест для режима sea
    tags = get_default_tags("sea")
    assert tags == {"harbour": True, "man_made": ["pier", "dock"]}
    
    # Тест для режима rail
    tags = get_default_tags("rail")
    assert tags == {"railway": ["station", "yard", "cargo_terminal"]}
    
    # Тест для неизвестного режима
    with pytest.raises(ValueError):
        get_default_tags("unknown")
    
    # Тест с разным регистром
    tags_lower = get_default_tags("AUTO")
    tags_upper = get_default_tags("auto")
    assert tags_lower == tags_upper


# Тесты для основных функций
def test_extract_coordinates():
    # Создаём тестовый GeoDataFrame
    points = [Point(30.3, 59.9), Point(30.4, 59.8)]
    gdf = gpd.GeoDataFrame({
        'name': ['Point1', 'Point2'],
        'building': ['warehouse', 'depot']
    }, geometry=points)
    
    coords_df = extract_coordinates(gdf)
    
    assert len(coords_df) == 2
    assert coords_df.iloc[0]['lat'] == 59.9
    assert coords_df.iloc[0]['lon'] == 30.3
    assert coords_df.iloc[1]['lat'] == 59.8
    assert coords_df.iloc[1]['lon'] == 30.4
    
    # Проверяем, что теги сохраняются
    assert coords_df.iloc[0]['tags']['name'] == 'Point1'


def test_extract_coordinates_with_polygons():
    # Тест с полигонами (центроиды)
    polygon1 = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    polygon2 = Polygon([(30.2, 59.2), (30.3, 59.2), (30.3, 59.3), (30.2, 59.3)])
    gdf = gpd.GeoDataFrame({
        'name': ['Poly1', 'Poly2'],
        'building': ['warehouse', 'depot']
    }, geometry=[polygon1, polygon2])
    
    coords_df = extract_coordinates(gdf)
    
    # Центроиды должны быть примерно в центре полигонов
    assert len(coords_df) == 2
    # Для первого полигона центроид должен быть близок к (30.05, 59.05)
    assert abs(coords_df.iloc[0]['lat'] - 59.05) < 0.01
    assert abs(coords_df.iloc[0]['lon'] - 30.05) < 0.01


@patch('services.logistics.nx.minimum_spanning_tree')
def test_build_mst_graph(mock_mst):
    # Создаём тестовый граф
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, weight=100)
    G.add_edge(1, 2, weight=200)
    G.add_edge(0, 2, weight=150)
    
    # Мокаем результат MST
    mst_result = nx.Graph()
    mst_result.add_nodes_from([0, 1, 2])
    mst_result.add_edge(0, 1, weight=100)
    mst_result.add_edge(0, 2, weight=150)
    mock_mst.return_value = mst_result
    
    result = build_mst_graph(G)
    
    # Проверяем, что вызвалась функция networkx
    mock_mst.assert_called_once()
    assert len(result.edges()) == 2


def test_build_geodesic_graph():
    # Создаём тестовый DataFrame с координатами
    coords_df = pd.DataFrame({
        'lat': [59.9, 59.8, 59.7],
        'lon': [30.3, 30.4, 30.5],
        'tags': [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}]
    })
    
    G = build_geodesic_graph(coords_df)
    
    # В графе должно быть 3 узла
    assert len(G.nodes()) == 3
    
    # Должно быть 3 ребра (полный граф из 3 узлов: 3*(3-1)/2 = 3)
    assert len(G.edges()) == 3
    
    # Проверяем, что веса рёбер соответствуют гаверсинусу
    edges = list(G.edges(data=True))
    for u, v, data in edges:
        expected_dist = haversine(
            coords_df.iloc[u]['lat'], 
            coords_df.iloc[u]['lon'],
            coords_df.iloc[v]['lat'], 
            coords_df.iloc[v]['lon']
        )
        assert abs(data['weight'] - expected_dist) < 0.001


def test_generate_logistics_mst_empty_gdf():
    # Тест, когда gdf пустой
    bbox = (29.81, 59.87, 29.88, 59.89)
    
    with patch('services.logistics.load_logistics_features') as mock_load:
        mock_load.return_value = gpd.GeoDataFrame()
        
        result = generate_logistics_mst(bbox, mode="auto")
        
        assert result['status'] == 'no_data'
        assert 'message' in result


def test_generate_logistics_mst_normal_case():
    # Тест нормального сценария
    bbox = (29.81, 59.87, 29.88, 59.89)
    
    # Создаём фейковый GeoDataFrame
    points = [Point(30.3, 59.9), Point(30.4, 59.8)]
    gdf = gpd.GeoDataFrame({
        'name': ['Point1', 'Point2'],
        'building': ['warehouse', 'depot']
    }, geometry=points)
    
    with patch('services.logistics.load_logistics_features') as mock_load, \
         patch('services.logistics.build_mst_graph') as mock_mst, \
         patch('services.logistics.visualize_mst_map') as mock_visualize:
        
        mock_load.return_value = gdf
        
        # Мокаем MST
        mst_graph = nx.Graph()
        mst_graph.add_nodes_from([0, 1])
        mst_graph.add_edge(0, 1, weight=1000.0)
        mock_mst.return_value = mst_graph
        
        # Мокаем визуализацию
        mock_visualize.return_value = "test_map.html"
        
        result = generate_logistics_mst(bbox, mode="auto")
        
        # Проверяем структуру результата
        assert result['status'] == 'ok'
        assert result['nodes_count'] == 2
        assert result['edges_count'] == 1
        assert result['total_distance'] == 1000.0
        assert len(result['points']) == 2
        assert len(result['edges']) == 1
        assert result['mode'] == 'auto'
        assert result['bbox'] == bbox


def test_generate_logistics_mst_cache_dir_creation():
    # Тест создания директории кэша
    bbox = (29.81, 59.87, 29.88, 59.89)
    
    # Создаём временные директории
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_subdir = os.path.join(temp_dir, "test_cache")
        
        # Создаём фейковый GeoDataFrame
        points = [Point(30.3, 59.9)]
        gdf = gpd.GeoDataFrame({
            'name': ['Point1'],
            'building': ['warehouse']
        }, geometry=points)
        
        with patch('services.logistics.load_logistics_features') as mock_load, \
             patch('services.logistics.build_mst_graph'), \
             patch('services.logistics.visualize_mst_map'):
            
            mock_load.return_value = gdf
            
            # Проверяем, что директория создаётся
            assert not os.path.exists(cache_subdir)
            
            generate_logistics_mst(bbox, mode="auto", cache_dir=cache_subdir)
            
            assert os.path.exists(cache_subdir)


def test_visualize_mst_map_output():
    # Тест визуализации
    bbox = (29.81, 59.87, 29.88, 59.89)
    
    coords_df = pd.DataFrame({
        'lat': [59.9, 59.8],
        'lon': [30.3, 30.4],
        'tags': [{'name': 'Point1'}, {'name': 'Point2'}]
    })
    
    mst = nx.Graph()
    mst.add_nodes_from([0, 1])
    mst.add_edge(0, 1, weight=1000.0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_map.html")
        
        result_path = visualize_mst_map(coords_df, mst, bbox, output_file)
        
        # Проверяем, что файл был создан
        assert os.path.exists(result_path)
        assert result_path == output_file


# Тесты для граничных случаев
def test_generate_logistics_mst_nan_handling():
    # Тест обработки NaN значений в координатах
    bbox = (29.81, 59.87, 29.88, 59.89)
    
    points = [Point(30.3, 59.9), Point(30.4, 59.8)]
    gdf = gpd.GeoDataFrame({
        'name': ['Point1', 'Point2'],
        'building': ['warehouse', None]  # None значение
    }, geometry=points)
    
    coords_df = extract_coordinates(gdf)
    
    # Проверяем, что обработка не падает
    G = build_geodesic_graph(coords_df)
    mst = build_mst_graph(G)
    
    # Проверяем, что все координаты действительны
    for _, row in coords_df.iterrows():
        assert not pd.isna(row['lat'])
        assert not pd.isna(row['lon'])


if __name__ == "__main__":
    pytest.main([__file__])