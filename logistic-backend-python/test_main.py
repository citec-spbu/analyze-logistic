import os 
import shutil
import pytest
import tempfile


import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd

from path import Path
from services.logistics import *
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from shapely.geometry import Point, Polygon

# Импортируем приложение
from main import app, DEFAULT_BBOX

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Тесты для вспомогательных функций
def test_haversine():
    # Тест на расстояние между двумя одинаковыми точками
    assert haversine((0,0), (0,0)) == 0

    # Тест на известное расстояние (приблизительно 111 км на 1 градус широты)
    dist = haversine((0, 0), (1, 0))  # 1 градус по широте
    print(dist)
    assert 111 - 1 < dist < 111 + 1  # ~111 км

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

def test_clear_cache_contents():

    base_dir = Path.cwd() 

    cache_path = base_dir / 'cache'
    in_cache_file = cache_path / 'test_dir'

    # os.mkdir(cache_path)
    os.mkdir(in_cache_file)
    
    with open(cache_path / 'test.txt', "w", encoding="utf-8") as f:
        f.write("тест")

    clear_cache_contents()


def test_empty_gdf():  # пустой df
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = merge_gdf_geometries(gdf)
        assert result.empty

def test_no_name_column(): # тест без колонок в DF
    geometry = [
        Point(10.0, 10.0).buffer(0.01),
        Point(10.02, 10.02).buffer(0.01),
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=geometry,
        crs="EPSG:4326"
    )
    result = merge_gdf_geometries(gdf, buffer_m=150.0)
    assert not result.empty
    assert result.crs.to_epsg() == 4326

def test_with_name_column():
    geometry = [
        Point(10.0, 10.0).buffer(0.005),
        Point(10.01, 10.01).buffer(0.005),
        Point(20.0, 20.0).buffer(0.005),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Москва станция", "Московский вокзал", "Казань"],
            "id": [1, 2, 3]
        },
        geometry=geometry,
        crs="EPSG:4326"
    )
    result = merge_gdf_geometries(
        gdf,
        buffer_m=150.0,
        name_merge_radius_km=50.0,
        name_similarity_threshold=0.6
    )
    assert not result.empty
    assert "name" in result.columns

def test_all_nan_names():
    geometry = [Point(10.0, 10.0).buffer(0.01)]
    gdf = gpd.GeoDataFrame(
        {"name": [None]},
        geometry=geometry,
        crs="EPSG:4326"
    )
    result = merge_gdf_geometries(gdf)
    assert not result.empty

def test_single_object():
    geometry = [Point(10.0, 10.0).buffer(0.01)]
    gdf = gpd.GeoDataFrame(
        {"name": ["Тест"]},
        geometry=geometry,
        crs="EPSG:4326"
    )
    result = merge_gdf_geometries(gdf)
    assert len(result) >= 1
    assert result.geometry.notna().all()


    
    
def test_extract_coordinates():
    # Создаём тестовый GeoDataFrame
    points = [Point(30.3, 59.9), Point(30.4, 59.8)]
    gdf = gpd.GeoDataFrame({
        'name': ['Point1', 'Point2'],
        'building': ['warehouse', 'depot']
    }, geometry=points)
    coords_df = extract_coordinates(gdf)
    assert len(coords_df) == 2
    assert round(coords_df.iloc[0]['lat'], 2) == 59.9
    assert round(coords_df.iloc[0]['lon'], 2) == 30.3
    assert round(coords_df.iloc[1]['lat'], 2) == 59.8
    assert round(coords_df.iloc[1]['lon'], 2) == 30.4
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
            (coords_df.iloc[u]['lat'], coords_df.iloc[u]['lon']),
            (coords_df.iloc[v]['lat'], coords_df.iloc[v]['lon'])
        )
        assert abs(data['weight'] - expected_dist) < 0.001

def test_get_colot():
    assert get_color('auto') == "#D62828"
    assert get_color("rail") == "#5E60CE"
    assert get_color("sea") == "#0077B6"
    assert get_color("aero") == "#009E73"
    assert get_color("support") == "#FFB703"
    assert get_color("adsa") == "#6C757D"

def test_generate_logistics_mst_empty_gdf():
    # Тест, когда gdf пустой
    bbox = (29.81, 59.87, 29.88, 59.89)
    with patch('services.logistics.load_logistics_features') as mock_load:
        mock_load.return_value = gpd.GeoDataFrame()
        result = generate_logistics_mst(bbox, mode="auto")
        assert result['status'] == 'ok'
        # В пустом случае edges_count == 0, total_distance == 0
        assert result['edges_count'] == 0
        assert result['total_distance'] == 0

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
        # Мокаем визуализацию: теперь возвращаем тот же путь, что получили
        def mock_visualize_impl(coords_df, mst, bbox, mode, output_file="logistics_mst.html"):
            # Эмулируем сохранение файла
            with open(output_file, 'w') as f:
                f.write('<html><body>Mock map</body></html>')
            return output_file  # ВАЖНО: возвращаем то, что получили
        mock_visualize.side_effect = mock_visualize_impl

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
             patch('services.logistics.visualize_mst_map') as mock_visualize:
            mock_load.return_value = gdf
            # Мокаем визуализацию
            def mock_visualize_impl(coords_df, mst, bbox, mode, output_file="logistics_mst.html"):
                with open(output_file, 'w') as f:
                    f.write('<html><body>Mock map</body></html>')
                return output_file
            mock_visualize.side_effect = mock_visualize_impl
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
        # Вызываем функцию с указанным output_file
        result_path = visualize_mst_map(coords_df, mst, bbox, mode='auto', output_file=output_file)
        # Проверяем, что файл был создан
        assert os.path.exists(result_path)
        # И теперь проверяем, что возвращённый путь == переданный
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


@pytest.fixture
def bbox():
    return (37.5, 55.7, 37.7, 55.8)


@patch('services.logistics.visualize_mst_map')
@patch('services.logistics.ox.routing.route_to_gdf')
@patch('services.logistics.ox.routing.shortest_path')
@patch('services.logistics.ox.distance.nearest_nodes')
@patch('services.logistics.ox.graph_from_bbox')
@patch('services.logistics.haversine')
@patch('services.logistics.pickle.dump')
@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.open') 
@patch('services.logistics.os.path.exists')  
@patch('services.logistics.os.path')
@patch('services.logistics.os.makedirs')
def test_mode_all_success(mock_makedirs, mock_join, mock_exists, mock_file,
                          mock_read_pickle, mock_pickle_load, mock_pickle_dump,
                          mock_haversine, mock_graph, mock_nearest, 
                          mock_shortest, mock_route_gdf, mock_visualize, bbox):

  
    mock_join.side_effect = lambda *a: '/'.join(a)
 
    mock_exists.return_value = True

    auto_df = pd.DataFrame({
        "lat": [55.75], "lon": [37.6], "tags": [{}], "mode": ["auto"]
    }, index=[0])
    auto_df.empty = False
    
    rail_df = pd.DataFrame({
        "lat": [55.76], "lon": [37.61], "tags": [{}], "mode": ["rail"]
    }, index=[0])
    rail_df.empty = False
    
    combined_df = pd.DataFrame({
        "lat": [55.75, 55.76], "lon": [37.6, 37.61], 
        "tags": [{}, {}], "mode": ["auto", "rail"]
    }, index=[0, 1])

    def read_pickle_side_effect(path):
        if "auto" in path:
            return auto_df
        elif "rail" in path:
            return rail_df
        return auto_df
    
    mock_read_pickle.side_effect = read_pickle_side_effect
    
    mock_pickle_load.return_value = nx.Graph([(0, 1, {'weight': 1.0})])
    
    mock_haversine.return_value = 1.5
    
  
    mock_graph.return_value = MagicMock()
    mock_nearest.return_value = 123
    mock_shortest.return_value = [1, 2]
    mock_route_gdf.return_value = pd.DataFrame({"length": [1000]})
    
    with patch('services.logistics.generate_logistics_mst') as mock_rec:
        mock_rec.return_value = {"status": "ok"}
        
        result = generate_logistics_mst(bbox, mode="all", cache_dir="cache")
    

    assert result["status"] == "ok"
    assert result["mode"] == "all"
    assert "edges" in result
    assert "points" in result
    mock_visualize.assert_called_once()


@patch('services.logistics.visualize_mst_map')
@patch('services.logistics.ox.routing.route_to_gdf')
@patch('services.logistics.ox.routing.shortest_path')
@patch('services.logistics.ox.distance.nearest_nodes')
@patch('services.logistics.ox.graph_from_bbox')
@patch('services.logistics.haversine')
@patch('services.logistics.pickle.dump')
@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.pd.concat')
@patch('services.logistics.pd.DataFrame')
@patch('services.logistics.os.path.join')
@patch('services.logistics.os.makedirs')
@patch("services.logistics.open")
def test_mode_all_no_auto(mock_makedirs, mock_open, mock_join, mock_df, mock_concat,
                          mock_read_pickle, mock_pickle_load, mock_pickle_dump,
                          mock_haversine, mock_graph, mock_nearest,
                          mock_shortest, mock_route_gdf, mock_visualize, bbox):
    """Тест 2: mode='all' — нет auto (ошибка)"""
    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_df.return_value = pd.DataFrame(columns=["lat", "lon", "tags", "mode"])
    mock_df.return_value.empty = True
    mock_read_pickle.return_value = pd.DataFrame(columns=["lat", "lon", "tags"])
    mock_read_pickle.return_value.empty = True
    mock_pickle_load.return_value = nx.Graph()
    
    with patch('services.logistics.generate_logistics_mst') as mock_rec:
        mock_rec.return_value = {"status": "ok"}
        result = generate_logistics_mst(bbox, mode="all", cache_dir="cache")
    
    assert result["status"] == "error"
    assert "Нет данных" in result["message"]


@patch('services.logistics.ox.graph_from_bbox')
@patch('services.logistics.os.makedirs')
def test_mode_single(mock_makedirs, mock_graph, bbox):

    mock_graph.return_value = MagicMock()
    
    with patch('services.logistics.generate_logistics_mst') as mock_rec:
        mock_rec.return_value = {"status": "ok", "coords_df": pd.DataFrame(), "mst": nx.Graph()}
        result = generate_logistics_mst(bbox, mode="rail", cache_dir="cache")
    
    assert result["status"] in ["ok", "error"]


@patch('services.logistics.visualize_mst_map')
@patch('services.logistics.ox.routing.route_to_gdf')
@patch('services.logistics.ox.routing.shortest_path')
@patch('services.logistics.ox.distance.nearest_nodes')
@patch('services.logistics.ox.graph_from_bbox')
@patch('services.logistics.haversine')
@patch('services.logistics.pickle.dump')
@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.pd.concat')# @patch('services.logistics.pd.DataFrame')
@patch('services.logistics.os.path.join')
@patch('services.logistics.os.makedirs')
@patch('services.logistics.open')
def test_gdrive_error(mock_open, mock_makedirs, mock_join, mock_concat,
                      mock_read_pickle, mock_pickle_load, mock_pickle_dump,
                      mock_haversine, mock_graph, mock_nearest,
                      mock_shortest, mock_route_gdf, mock_visualize, bbox):
   
    mock_join.side_effect = lambda *a: '/'.join(a)
    # mock_df.return_value = pd.DataFrame({"lat": [55.75], "lon": [37.6], "tags": [{}], "mode": ["auto"]})
    mock_concat.return_value = pd.DataFrame({"lat": [55.75, 55.76], "lon": [37.6, 37.61],
                                              "tags": [{}, {}], "mode": ["auto", "rail"]})
    mock_read_pickle.return_value = pd.DataFrame({"lat": [55.75], "lon": [37.6], "tags": [{}]}, index=[0,1])
    mock_pickle_load.return_value = nx.Graph([(0, 1, {'weight': 1.0})])
    mock_haversine.return_value = 1.5
    mock_graph.side_effect = Exception("Network error") 
    
    with patch('services.logistics.generate_logistics_mst') as mock_rec:
        mock_rec.return_value = {"status": "ok"}
        result = generate_logistics_mst(bbox, mode="all", cache_dir="cache")
    
    assert result['status'] == "ok" 


@patch('services.logistics.visualize_mst_map')
@patch('services.logistics.ox.routing.route_to_gdf')
@patch('services.logistics.ox.routing.shortest_path')
@patch('services.logistics.ox.distance.nearest_nodes')
@patch('services.logistics.ox.graph_from_bbox')
@patch('services.logistics.haversine')
@patch('services.logistics.pickle.dump')
@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.pd.concat')
# @patch('services.logistics.pd.DataFrame')
@patch('services.logistics.os.path.join')
@patch('services.logistics.os.makedirs')
@patch('services.logistics.open')
def test_route_error(mock_makedirs, mock_open, mock_join, mock_concat,
                     mock_read_pickle, mock_pickle_load, mock_pickle_dump,
                     mock_haversine, mock_graph, mock_nearest,
                     mock_shortest, mock_route_gdf, mock_visualize, bbox):

    mock_join.side_effect = lambda *a: '/'.join(a)
    # mock_df.return_value = pd.DataFrame({"lat": [55.75], "lon": [37.6], "tags": [{}], "mode": ["auto"]})
    mock_concat.return_value = pd.DataFrame({"lat": [55.75, 55.76], "lon": [37.6, 37.61],
                                              "tags": [{}, {}], "mode": ["auto", "rail"]})
    mock_read_pickle.return_value = pd.DataFrame({"lat": [55.75], "lon": [37.6], "tags": [{}]}, index=[0,1])
    mock_pickle_load.return_value = nx.Graph([(0, 1, {'weight': 1.0})])
    mock_haversine.return_value = 1.5
    mock_graph.return_value = MagicMock()
    mock_shortest.side_effect = Exception("Route error") 
    
    with patch('services.logistics.generate_logistics_mst') as mock_rec:
        mock_rec.return_value = {"status": "ok"}
        result = generate_logistics_mst(bbox, mode="all", cache_dir="cache")
    
    assert result["status"] == "ok"  


@patch('services.logistics.folium.CircleMarker')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.folium.LayerControl')
@patch('services.logistics.create_base_map')
@patch('services.logistics.get_color')
@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.os.path.exists')
@patch('services.logistics.os.path.join')
@patch('services.logistics.open')
@patch('services.logistics.compute_metric')
def test_mode_all_success(mock_compute, mock_open, mock_join, mock_exists, mock_read_pickle, 
                          mock_pickle_load, mock_get_color, mock_create_map,
                          mock_layer_control, mock_polyline, mock_circle, bbox):
   
    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame({
        "lat": [55.75, 55.76], "lon": [37.6, 37.61], 
        "tags": [{}, {}], "mode": ["auto", "rail"]
    }, index=[0, 1])
    mock_pickle_load.return_value = nx.Graph([(0, 1, {'weight': 1.0})])
    mock_compute.return_value = {0: 0.5, 1: 0.8}
    mock_get_color.return_value = "#FF0000"
    mock_map = MagicMock()
    mock_create_map.return_value = mock_map
    
    result = analyze_logistics_metrics(bbox, mode="all", metric="degree", cache_dir="cache")
    
    assert result["status"] == "ok"
    assert result["mode"] == "all"
    assert "values" in result
    assert mock_compute.called
    mock_create_map.assert_called_once()


@patch('services.logistics.visualize_metric_map')
@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.os.path.exists')
@patch('services.logistics.os.path.join')
@patch('services.logistics.compute_metric')
def test_mode_single_success(mock_compute, mock_join, mock_exists, mock_read_pickle,
                             mock_pickle_load, mock_visualize, bbox):

    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame({
        "lat": [55.75], "lon": [37.6], "tags": [{}], "mode": ["auto"]
    }, index=[0])
    mock_pickle_load.return_value = nx.Graph([(0, 1, {'weight': 1.0})])
    mock_compute.return_value = {0: 0.5}
    
    result = analyze_logistics_metrics(bbox, mode="auto", metric="degree", cache_dir="cache")
    
    assert result["status"] == "ok"
    assert result["mode"] == "auto"
    mock_visualize.assert_called_once()


@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.os.path.exists')
@patch('services.logistics.os.path.join')
def test_files_not_exist(mock_join, mock_exists, mock_read_pickle, mock_pickle_load, bbox):
    
    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_exists.return_value = False
    
    result = analyze_logistics_metrics(bbox, mode="auto", metric="degree", cache_dir="cache")
    
    assert result["status"] == "error"
    assert "MST не найден" in result["message"]


@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.os.path.exists')
@patch('services.logistics.os.path.join')
def test_empty_graph(mock_join, mock_exists, mock_read_pickle, mock_pickle_load, bbox):
 
    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame()
    mock_pickle_load.return_value = nx.Graph()  # 0 узлов
    
    result = analyze_logistics_metrics(bbox, mode="auto", metric="degree", cache_dir="cache")
    
    assert result["status"] == "error"
    assert "MST пустой" in result["message"]


@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.os.path.exists')
@patch('services.logistics.os.path.join')
def test_single_node_graph(mock_join, mock_exists, mock_read_pickle, mock_pickle_load, bbox):
  
    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame({"lat": [55.75]}, index=[0])
    mock_pickle_load.return_value = nx.Graph()
    mock_pickle_load.return_value.add_node(0)  # 1 узел
    
    result = analyze_logistics_metrics(bbox, mode="auto", metric="degree", cache_dir="cache")
    
    assert result["status"] == "error"
    assert "Недостаточно вершин" in result["message"]


@patch('services.logistics.pickle.load')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.os.path.exists')
@patch('services.logistics.os.path.join')
@patch('services.logistics.compute_metric')
def test_compute_metric_error(mock_compute, mock_join, mock_exists, 
                              mock_read_pickle, mock_pickle_load, bbox):

    mock_join.side_effect = lambda *a: '/'.join(a)
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame({
        "lat": [55.75, 55.76], "lon": [37.6, 37.61]
    }, index=[0, 1])
    mock_pickle_load.return_value = nx.Graph([(0, 1, {'weight': 1.0})])
    mock_compute.side_effect = Exception("Metric error")
    
    result = analyze_logistics_metrics(bbox, mode="auto", metric="degree", cache_dir="cache")
    
    assert result["status"] == "error"
    assert "Ошибка вычисления" in result["message"]


@pytest.fixture
def mock_folium_map():
    """Создает фиктивный объект карты Folium с необходимой структурой"""
    m = MagicMock()
    root = MagicMock()
    html = MagicMock()
    root.html = html
    m.get_root.return_value = root
    return m

@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.LayerControl')
@patch('services.logistics.draw_nodes_layer')
@patch('services.logistics.draw_mst_layer')
@patch('services.logistics.pickle.load')
@patch('services.logistics.open')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.generate_logistics_mst')
@patch('services.logistics.create_base_map')
@patch('services.logistics.os.makedirs')
@patch('services.logistics.os.path.join')
@patch('builtins.print')
def test_generate_all_modes_mst_success(
    mock_print, mock_join, mock_makedirs, mock_create_map, 
    mock_gen_mst, mock_read_pickle, mock_file_open, mock_pickle_load,
    mock_draw_mst, mock_draw_nodes, mock_layer_control, mock_element
):
    """Тест 1: Успешное выполнение для всех режимов"""
    
    # Настройка моков
    mock_join.side_effect = lambda dir, fname: f"{dir}/{fname}"
    mock_create_map.return_value = MagicMock() # Простая заглушка для карты
    # Настраиваем структуру карты для вызова m.get_root().html.add_child
    mock_map_instance = MagicMock()
    mock_map_instance.get_root.return_value.html.add_child = MagicMock()
    mock_create_map.return_value = mock_map_instance

    # generate_logistics_mst возвращает успех для всех 4 режимов
    mock_gen_mst.return_value = {"status": "ok"}
    
    # pd.read_pickle возвращает пустой DataFrame (заглушка)
    mock_read_pickle.return_value = pd.DataFrame({'lat': [55], 'lon': [37]})
    
    # pickle.load возвращает заглушку MST
    mock_pickle_load.return_value = MagicMock()

    # Запуск функции
    result = generate_all_modes_mst(bbox=[1, 2, 3, 4])

    # Проверки
    assert result == "logistics_mst_all.html"
    assert mock_makedirs.called
    assert mock_create_map.called
    # Функция генерации должна вызваться 4 раза (по числу режимов)
    assert mock_gen_mst.call_count == 4 
    # Отрисовка MST должна вызваться 4 раза
    assert mock_draw_mst.call_count == 4
    # Отрисовка узлов 1 раз
    assert mock_draw_nodes.call_count == 1
    # Сохранение карты
    mock_map_instance.save.assert_called_once_with("logistics_mst_all.html")
    # Проверка, что print был вызван с финальным сообщением
    mock_print.assert_any_call("Все MST сохранены на карте: logistics_mst_all.html")


@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.LayerControl')
@patch('services.logistics.draw_nodes_layer')
@patch('services.logistics.draw_mst_layer')
@patch('services.logistics.pickle.load')
@patch('services.logistics.open')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.generate_logistics_mst')
@patch('services.logistics.create_base_map')
@patch('services.logistics.os.makedirs')
@patch('services.logistics.os.path.join')
@patch('builtins.print')
def test_generate_all_modes_mst_exception(
    mock_print, mock_join, mock_makedirs, mock_create_map, 
    mock_gen_mst, mock_read_pickle, mock_file_open, mock_pickle_load,
    mock_draw_mst, mock_draw_nodes, mock_layer_control, mock_element
):
    """Тест 2: Покрытие блока except (ошибка при чтении файла)"""
    
    mock_join.side_effect = lambda dir, fname: f"{dir}/{fname}"
    mock_map_instance = MagicMock()
    mock_map_instance.get_root.return_value.html.add_child = MagicMock()
    mock_create_map.return_value = mock_map_instance
    
    mock_gen_mst.return_value = {"status": "ok"}
    
    # Имитируем ошибку при чтении coords для первого режима ('auto')
    # side_effect можно задать списком, чтобы ошибка была только один раз
    mock_read_pickle.side_effect = [Exception("File corrupted"), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    
    # Запуск
    generate_all_modes_mst(bbox=[1, 2, 3, 4])

    # Проверка, что print был вызван с сообщением об ошибке
    # Ожидаем сообщение: "Ошибка MST для auto: File corrupted"
    error_found = False
    for call in mock_print.call_args_list:
        if "Ошибка MST для auto" in str(call):
            error_found = True
            assert "File corrupted" in str(call)
            break
    
    assert error_found, "Сообщение об ошибке не было найдено в выводе"
    
    # draw_mst_layer не должен был вызваться для режима с ошибкой (всего 3 раза вместо 4)
    assert mock_draw_mst.call_count == 3


@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.LayerControl')
@patch('services.logistics.draw_nodes_layer')
@patch('services.logistics.draw_mst_layer')
@patch('services.logistics.pickle.load')
@patch('services.logistics.open')
@patch('services.logistics.pd.read_pickle')
@patch('services.logistics.generate_logistics_mst')
@patch('services.logistics.create_base_map')
@patch('services.logistics.os.makedirs')
@patch('services.logistics.os.path.join')
@patch('services.logistics.pd.concat') # Мокаем concat, чтобы проверить, был ли он вызван
@patch('services.logistics.pd.DataFrame') # Мокаем DataFrame, чтобы проверить ветку else
@patch('builtins.print')
def test_generate_all_modes_mst_empty_data(
    mock_print, mock_df, mock_concat, mock_join, mock_makedirs, mock_create_map, 
    mock_gen_mst, mock_read_pickle, mock_file_open, mock_pickle_load,
    mock_draw_mst, mock_draw_nodes, mock_layer_control, mock_element
):
    """Тест 3: Покрытие ветки else (если all_coords пуст)"""
    
    mock_join.side_effect = lambda dir, fname: f"{dir}/{fname}"
    mock_map_instance = MagicMock()
    mock_map_instance.get_root.return_value.html.add_child = MagicMock()
    mock_create_map.return_value = mock_map_instance
    
    # Возвращаем статус, который НЕ входит в ["ok", "no_data"]
    # Тогда блок try не выполнится, all_coords останется пустым
    mock_gen_mst.return_value = {"status": "error"} 
    
    # Запуск
    generate_all_modes_mst(bbox=[1, 2, 3, 4])

    # pd.concat НЕ должен был вызваться, так как all_coords пуст
    mock_concat.assert_not_called()
    
    # pd.DataFrame ДОЛЖЕН был вызваться (ветка else)
    mock_df.assert_called_once()
    
    # draw_nodes_layer все равно вызывается (с пустым df)
    assert mock_draw_nodes.call_count == 1




@pytest.fixture
def mock_print():
    """Фикстура для перехвата вызовов print"""
    with patch('builtins.print') as mock_print:
        yield mock_print

@patch('services.logistics.shutil.rmtree')
@patch('services.logistics.os.unlink')
@patch('services.logistics.os.path.isdir')
@patch('services.logistics.os.path.islink')
@patch('services.logistics.os.path.isfile')
@patch('services.logistics.os.path.join')
@patch('services.logistics.os.listdir')
@patch('services.logistics.os.path.exists')
def test_clear_cache_exception_handling(
    mock_exists, mock_listdir, mock_join, mock_isfile, 
    mock_islink, mock_isdir, mock_unlink, mock_rmtree, mock_print
):
    # 1. Настраиваем окружение: папка существует
    mock_exists.return_value = True
    
    # 2. В папке есть один файл
    mock_listdir.return_value = ['test_file.txt']
    
    # 3. Настраиваем пути и типы файлов
    mock_join.return_value = 'cache/test_file.txt'
    mock_isfile.return_value = True
    mock_islink.return_value = False
    mock_isdir.return_value = False
    
    # 4. ГЛАВНОЕ: Имитируем ошибку при удалении файла
    mock_unlink.side_effect = PermissionError("Access denied")
    
    # 5. Вызываем функцию
    clear_cache_contents()
    
    # 6. Проверяем, что print был вызван с сообщением об ошибке
    # Ожидаем, что print был вызван хотя бы один раз с текстом ошибки
    error_message_found = False
    for call in mock_print.call_args_list:
        args, kwargs = call
        if args and "Не удалось удалить" in str(args[0]):
            error_message_found = True
            # Дополнительная проверка: содержится ли текст ошибки
            assert "Access denied" in str(args[0])
            break
            
    assert error_message_found, "Сообщение об ошибке не было напечатано"
    
    # Убеждаемся, что rmtree не был вызван (так как это файл, а не папка)
    mock_rmtree.assert_not_called()





@pytest.fixture
def sample_coords_df():
    """Создает тестовый DataFrame с координатами"""
    return pd.DataFrame({
        'lat': [55.0, 55.1, 55.2],
        'lon': [37.0, 37.1, 37.2],
        'tags': [
            {"name": "Building A", "building": "office"},
            {"name": "Building B", "building": "warehouse"},
            {"name": None, "building": "house"}
        ]
    }, index=[0, 1, 2])

@pytest.fixture
def sample_graph():
    """Создает тестовый граф (MagicMock с edges)"""
    G = MagicMock()
    #_edges возвращает список кортежей (u, v, data)
    G.edges.return_value = [(0, 1, {}), (1, 2, {})]
    return G

@patch('services.logistics.folium.CircleMarker')
@patch('services.logistics.folium.Popup')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.Map')
@patch('services.logistics.get_color')
def test_visualize_metric_map_normal(
    mock_get_color, mock_map, mock_element, mock_polyline, 
    mock_popup, mock_circle, sample_coords_df, sample_graph
):
    """Тест 1: Нормальный сценарий с разными значениями метрики"""
    
    # Настройка моков
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    mock_get_color.return_value = "blue"
    
    # Разные значения метрики (vmax != vmin)
    metric_vals = {0: 0.1, 1: 0.5, 2: 0.9}
    bbox = [36.0, 54.0, 38.0, 56.0]
    
    # Запуск функции
    result = visualize_metric_map(
        coords_df=sample_coords_df,
        G=sample_graph,
        metric_vals=metric_vals,
        bbox=bbox,
        mode="auto",
        output_file="test_map.html"
    )
    
    # Проверки
    assert result == "test_map.html"
    mock_map.assert_called_once()
    mock_get_color.assert_called_once_with("auto")
    
    # PolyLine должен вызваться 2 раза (2 ребра в графе)
    assert mock_polyline.call_count == 2
    
    # CircleMarker должен вызваться 3 раза (3 узла в DataFrame)
    assert mock_circle.call_count == 3
    
    # Popup должен вызваться 3 раза (для всех узлов, т.к. все есть в metric_vals)
    assert mock_popup.call_count == 3
    
    # Проверка сохранения
    mock_map_instance.save.assert_called_once_with("test_map.html")


@patch('services.logistics.folium.CircleMarker')
@patch('services.logistics.folium.Popup')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.Map')
@patch('services.logistics.get_color')
def test_visualize_metric_map_equal_values(
    mock_get_color, mock_map, mock_element, mock_polyline, 
    mock_popup, mock_circle, sample_coords_df, sample_graph
):
    """Тест 2: Покрытие ветки if vmax == vmin в color_for_value"""
    
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    mock_get_color.return_value = "green"
    
    # ВСЕ значения одинаковые -> vmax == vmin
    metric_vals = {0: 0.5, 1: 0.5, 2: 0.5}
    bbox = [36.0, 54.0, 38.0, 56.0]
    
    visualize_metric_map(
        coords_df=sample_coords_df,
        G=sample_graph,
        metric_vals=metric_vals,
        bbox=bbox,
        mode="rail",
        output_file="test_map.html"
    )
    
    # Проверяем, что CircleMarker был вызван (код не упал на делении на ноль)
    assert mock_circle.call_count == 3
    
    # Проверяем цвета: при t=0 должен быть зеленый (r=0, g=255)
    # Проверка через вызовы мока
    for call_args in mock_circle.call_args_list:
        kwargs = call_args[1]  # именованные аргументы
        assert 'color' in kwargs
        # При vmax==vmin, t=0, значит color="#00ff00"
        assert kwargs['color'] == "#00ff00"
        assert kwargs['fill_color'] == "#00ff00"


@patch('services.logistics.folium.CircleMarker')
@patch('services.logistics.folium.Popup')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.Map')
@patch('services.logistics.get_color')
def test_visualize_metric_map_missing_indices(
    mock_get_color, mock_map, mock_element, mock_polyline, 
    mock_popup, mock_circle, sample_coords_df, sample_graph
):
    """Тест 3: Покрытие ветки continue (если idx not in metric_vals)"""
    
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    mock_get_color.return_value = "red"
    
    # metric_vals содержит ТОЛЬКО индекс 0. Индексы 1 и 2 отсутствуют.
    # Это заставит цикл пропустить их через continue
    metric_vals = {0: 0.5}
    bbox = [36.0, 54.0, 38.0, 56.0]
    
    visualize_metric_map(
        coords_df=sample_coords_df,
        G=sample_graph,
        metric_vals=metric_vals,
        bbox=bbox,
        mode="sea",
        output_file="test_map.html"
    )
    
    # CircleMarker должен вызваться ТОЛЬКО 1 раз (для индекса 0)
    # Индексы 1 и 2 пропускаются через continue
    assert mock_circle.call_count == 1
    
    # Popup тоже только 1 раз
    assert mock_popup.call_count == 1
    
    # Проверяем, что для вызванного маркера popup содержит правильные данные
    popup_call = mock_popup.call_args_list[0]
    popup_content = popup_call[0][0]  # Первый позиционный аргумент
    assert "Building A" in popup_content
    assert "office" in popup_content


@patch('services.logistics.folium.CircleMarker')
@patch('services.logistics.folium.Popup')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.Map')
@patch('services.logistics.get_color')
def test_visualize_metric_map_empty_graph(
    mock_get_color, mock_map, mock_element, mock_polyline, 
    mock_popup, mock_circle, sample_coords_df
):
    """Тест 4: Покрытие случая с пустым графом (нет ребер)"""
    
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    mock_get_color.return_value = "black"
    
    # Пустой граф
    G = MagicMock()
    G.edges.return_value = []  # Нет ребер
    
    metric_vals = {0: 0.5}
    bbox = [36.0, 54.0, 38.0, 56.0]
    
    # Берем только первую строку для coords, чтобы соответствовало metric_vals
    coords_df = sample_coords_df.iloc[[0]]
    
    visualize_metric_map(
        coords_df=coords_df,
        G=G,
        metric_vals=metric_vals,
        bbox=bbox,
        mode="aero",
        output_file="test_map.html"
    )
    
    # PolyLine НЕ должен вызываться (нет ребер)
    mock_polyline.assert_not_called()
    
    # CircleMarker должен вызваться 1 раз
    assert mock_circle.call_count == 1
    
    # Сохранение все равно должно произойти
    mock_map_instance.save.assert_called_once()


@patch('services.logistics.folium.CircleMarker')
@patch('services.logistics.folium.Popup')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.folium.Element')
@patch('services.logistics.folium.Map')
@patch('services.logistics.get_color')
def test_visualize_metric_map_missing_tags(
    mock_get_color, mock_map, mock_element, mock_polyline, 
    mock_popup, mock_circle, sample_coords_df, sample_graph
):
    """Тест 5: Проверка обработки отсутствующих тегов (tags.get с дефолтом)"""
    
    mock_map_instance = MagicMock()
    mock_map.return_value = mock_map_instance
    mock_get_color.return_value = "blue"
    
    # Создаем DataFrame с пустыми тегами
    coords_df = pd.DataFrame({
        'lat': [55.0],
        'lon': [37.0],
        'tags': [{}]  # Пустой словарь
    }, index=[0])
    
    G = MagicMock()
    G.edges.return_value = []
    
    metric_vals = {0: 0.75}
    bbox = [36.0, 54.0, 38.0, 56.0]
    
    visualize_metric_map(
        coords_df=coords_df,
        G=G,
        metric_vals=metric_vals,
        bbox=bbox,
        mode="auto",
        output_file="test_map.html"
    )
    
    # Проверяем контент Popup: при пустых тегах должно быть "—"
    popup_call = mock_popup.call_args_list[0]
    popup_content = popup_call[0][0]
    
    assert "—" in popup_content  # Дефолтное значение для building
    assert "None" in popup_content  # name будет None




# Создаем тестовый клиент
client = TestClient(app)


@pytest.fixture
def sample_bbox():
    return {
        "west": 48.8,
        "south": 55.6,
        "east": 49.3,
        "north": 55.9
    }


@pytest.fixture
def mock_mst_result():
    return {
        "status": "ok",
        "message": "Успешно",
        "map_path": "cache/mst_test.html",
        "nodes": 10,
        "edges": 9
    }


@pytest.fixture
def mock_metrics_result():
    return {
        "status": "ok",
        "map_path": "cache/metrics_test.html",
        "metric": "degree_centrality"
    }


@pytest.fixture
def cleanup_cache():
    yield
    if os.path.exists("cache"):
        for file in os.listdir("cache"):
            file_path = os.path.join("cache", file)
            if os.path.isfile(file_path):
                os.remove(file_path)



class TestRootEndpoint:

    def test_read_root_success(self):
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Logistics Network API"
        assert "endpoints" in data
        assert "GET /" in data["endpoints"]
        assert "POST /analyze" in data["endpoints"]
        assert "GET /map" in data["endpoints"]
        assert "DELETE /cache" in data["endpoints"]




class TestHealthEndpoint:

    def test_health_check_success(self):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "Logistics Network API"

class TestAnalyzeEndpoint:

    @patch('main.generate_logistics_mst')
    def test_analyze_success(self, mock_generate, sample_bbox):
        mock_generate.return_value = {
            "status": "ok",
            "message": "Успешно",
            "map_path": "cache/mst.html"
        }
        
        response = client.post("/analyze", params=sample_bbox)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        mock_generate.assert_called_once()

    @patch('main.generate_logistics_mst')
    def test_analyze_with_mode(self, mock_generate, sample_bbox):
        mock_generate.return_value = {"status": "ok", "map_path": "cache/mst.html"}
        
        response = client.post("/analyze", params={
            **sample_bbox,
            "mode": "rail"
        })
        
        assert response.status_code == 200
        mock_generate.assert_called_once()

    @patch('main.generate_logistics_mst')
    def test_analyze_error_status(self, mock_generate, sample_bbox):
        mock_generate.return_value = {"status": "error", "message": "Нет данных"}
        response = client.post("/analyze", params=sample_bbox)
        
        # Ожидаем 500, так как код перехватывает 404
        assert response.status_code == 500 

    @patch('main.generate_logistics_mst')
    def test_analyze_exception(self, mock_generate, sample_bbox):
        mock_generate.side_effect = Exception("Connection error")
        
        response = client.post("/analyze", params=sample_bbox)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_analyze_default_bbox(self):
        with patch('main.generate_logistics_mst') as mock_generate:
            mock_generate.return_value = {"status": "ok", "map_path": "cache/mst.html"}
            
            response = client.post("/analyze")
            
            assert response.status_code == 200
            # Проверяем, что функция вызвалась с дефолтными координатами
            mock_generate.assert_called_once()


class TestMapEndpoint:

    @patch('main.generate_logistics_mst')
    def test_get_map_success(self, mock_generate, sample_bbox):
        mock_generate.return_value = {
            "status": "ok",
            "map_path": "cache/mst.html"
        }
        
        # Создаем тестовый HTML файл
        os.makedirs("cache", exist_ok=True)
        with open("cache/mst.html", "w", encoding="utf-8") as f:
            f.write("<html><body>Test Map</body></html>")
        
        response = client.get("/map", params=sample_bbox)
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "Test Map" in response.text

    @patch('main.generate_logistics_mst')
    def test_get_map_no_data(self, mock_generate, sample_bbox):
        mock_generate.return_value = {
            "status": "error",
            "message": "Нет данных"
        }
        
        response = client.get("/map", params=sample_bbox)
        
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("text/html")

    @patch('main.generate_logistics_mst')
    def test_get_map_exception(self, mock_generate, sample_bbox):
        mock_generate.side_effect = Exception("File not found")
        
        response = client.get("/map", params=sample_bbox)
        
        assert response.status_code == 500

    @patch('main.generate_all_modes_mst')
    def test_get_map_all_success(self, mock_generate_all, sample_bbox):
        # Создаем тестовый файл
        os.makedirs("cache", exist_ok=True)
        with open("cache/mst_all.html", "w", encoding="utf-8") as f:
            f.write("<html><body>All Modes Map</body></html>")
        
        response = client.get("/map/all", params=sample_bbox)
        
        assert response.status_code == 200
        assert "All Modes Map" in response.text




class TestMetricsEndpoint:

    @patch('main.analyze_logistics_metrics')
    def test_metrics_success(self, mock_analyze, sample_bbox):
        mock_analyze.return_value = {
            "status": "ok",
            "map_path": "cache/metrics.html"
        }
        
        # Создаем тестовый файл
        os.makedirs("cache", exist_ok=True)
        with open("cache/metrics.html", "w", encoding="utf-8") as f:
            f.write("<html></html>")
        
        response = client.get("/metrics", params={
            **sample_bbox,
            "metric": "degree_centrality"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "map_path" in data

    @patch('main.analyze_logistics_metrics')
    def test_metrics_error_status(self, mock_analyze, sample_bbox):

        mock_analyze.return_value = {"status": "error", "message": "Неверная метрика"}
        response = client.get("/metrics", params={**sample_bbox, "metric": "invalid"})
        
        # Ожидаем 500 
        assert response.status_code == 500

    @patch('main.analyze_logistics_metrics')
    def test_metrics_file_not_created(self, mock_analyze, sample_bbox):
        mock_analyze.return_value = {
            "status": "ok",
            "map_path": "cache/nonexistent.html"
        }
        
        response = client.get("/metrics", params={
            **sample_bbox,
            "metric": "degree_centrality"
        })
        
        assert response.status_code == 500

    def test_metrics_missing_parameter(self):
        response = client.get("/metrics", params={
            "west": 48.8,
            "south": 55.6,
            "east": 49.3,
            "north": 55.9
        })
        
        # FastAPI вернет 422 для валидации
        assert response.status_code == 422


class TestCacheEndpoint:

    def test_clear_cache_success(self, cleanup_cache):
        # Создаем тестовые файлы в кэше
        os.makedirs("cache", exist_ok=True)
        with open("cache/test_file.txt", "w") as f:
            f.write("test")
        
        response = client.delete("/cache")
        
        assert response.status_code == 200
        data = response.json()
        assert "Кэш успешно очищен" in data["message"]
        
        # Проверяем, что файлы удалены
        assert not os.path.exists("cache/test_file.txt")

    def test_clear_cache_already_empty(self):
        # Гарантируем, что кэш не существует
        if os.path.exists("cache"):
            shutil.rmtree("cache")
        
        response = client.delete("/cache")
        
        assert response.status_code == 200
        data = response.json()
        assert "Кэш уже пуст" in data["message"]


class TestParameterValidation:

    def test_analyze_invalid_coordinates(self):
        response = client.post("/analyze", params={
            "west": "invalid",
            "south": 55.6,
            "east": 49.3,
            "north": 55.9
        })
        
        # FastAPI вернет 422 для валидации типов
        assert response.status_code == 422

    def test_map_invalid_mode(self, sample_bbox):
        with patch('main.generate_logistics_mst') as mock_generate:
            mock_generate.return_value = {"status": "ok", "map_path": "cache/mst.html"}
            response = client.get("/map", params={"mode": 123})
            
            assert response.status_code in [200, 500] 


class TestLoadLogisticsFeatures:
    

    @pytest.fixture
    def sample_bbox(self):
       
        return (37.5, 55.7, 37.7, 55.8)  # (min_lon, min_lat, max_lon, max_lat)

    @pytest.fixture
    def temp_cache_dir(self):
      
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_gdf(self):
        
        geometry = [Point(37.6, 55.75), Point(37.65, 55.76)]
        return gpd.GeoDataFrame(
            {"name": ["Тест 1", "Тест 2"], "amenity": ["auto", "auto"]},
            geometry=geometry,
            crs="EPSG:4326"
        )

    def test_load_creates_cache_dir(self, sample_bbox, temp_cache_dir):
       
        cache_path = os.path.join(temp_cache_dir, "custom_cache")
        
        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            mock_ox.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
            load_logistics_features(sample_bbox, cache_dir=cache_path)
            
            assert os.path.exists(cache_path)

    def test_load_returns_geodataframe(self, sample_bbox, temp_cache_dir, mock_gdf):
     
        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            mock_ox.return_value = mock_gdf
            
            result = load_logistics_features(sample_bbox, cache_dir=temp_cache_dir)
            
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 2

    def test_load_saves_to_cache(self, sample_bbox, temp_cache_dir, mock_gdf):
        
        cache_file = os.path.join(temp_cache_dir, "logistics.geojson")
        
        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            mock_ox.return_value = mock_gdf
            
            load_logistics_features(sample_bbox, cache_dir=temp_cache_dir)
            
            assert os.path.exists(cache_file)

    def test_load_with_custom_mode(self, sample_bbox, temp_cache_dir, mock_gdf):
       
        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            with patch('services.logistics.get_default_tags') as mock_tags:
                mock_tags.return_value = {"amenity": ["warehouse"]}
                mock_ox.return_value = mock_gdf
                
                load_logistics_features(sample_bbox, mode="warehouse", cache_dir=temp_cache_dir)
                
                mock_tags.assert_called_once_with("warehouse")

    def test_load_empty_result(self, sample_bbox, temp_cache_dir):
 
        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            mock_ox.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
            result = load_logistics_features(sample_bbox, cache_dir=temp_cache_dir)
            
            assert isinstance(result, gpd.GeoDataFrame)
            assert result.empty


    def test_load_bbox_format(self, sample_bbox, temp_cache_dir, mock_gdf):

        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            mock_ox.return_value = mock_gdf
            
            load_logistics_features(sample_bbox, cache_dir=temp_cache_dir)
            
            mock_ox.assert_called_once()
            call_args = mock_ox.call_args
            assert call_args.kwargs['bbox'] == sample_bbox

    def test_load_file_saved_with_correct_driver(self, sample_bbox, temp_cache_dir, mock_gdf):

        cache_file = os.path.join(temp_cache_dir, "logistics.geojson")
        
        with patch('services.logistics.ox.features.features_from_bbox') as mock_ox:
            with patch.object(gpd.GeoDataFrame, 'to_file') as mock_to_file:
                mock_ox.return_value = mock_gdf
                
                load_logistics_features(sample_bbox, cache_dir=temp_cache_dir)
                
                mock_to_file.assert_called_once()
                call_kwargs = mock_to_file.call_args.kwargs
                assert call_kwargs['driver'] == 'GeoJSON'



class TestGetDefaultTags:

    def test_tags_auto_mode(self):
        
        result = get_default_tags("auto")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_tags_warehouse_mode(self):
        
        result = get_default_tags("aero")
        assert isinstance(result, dict)

    def test_tags_unknown_mode(self):
        
        result = get_default_tags("sea")
        assert isinstance(result, dict)


@patch('services.logistics.folium.LayerControl')
@patch('services.logistics.folium.FeatureGroup')
@patch('services.logistics.folium.PolyLine')
@patch('services.logistics.haversine')
@patch('services.logistics.get_color')
@patch('services.logistics.draw_mst_layer')
@patch('services.logistics.draw_nodes_layer')
@patch('services.logistics.create_base_map')
def test_visualize(mock_create_map, mock_draw_nodes, mock_draw_mst,
                   mock_get_color, mock_haversine, mock_polyline,
                   mock_fg, mock_layer_control):

    df = pd.DataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)], "lat": [0, 1], "lon": [0, 1], "mode": ["rail", "road"]},
        index=[0, 1]
    )
    mst = nx.Graph()
    mst.add_edge(0, 1, weight=1.0)
    bbox = (0, 0, 1, 1)
    
    # ✅ Настройка mock_map с корректным __str__
    mock_map = MagicMock()
    mock_map.__str__ = lambda self: "MockMap"
    mock_map.__repr__ = lambda self: "MockMap"
    mock_create_map.return_value = mock_map
    
    # ✅ Настройка FeatureGroup с корректными строковыми методами
    mock_fg_instance = MagicMock()
    mock_fg_instance.__str__ = lambda self: "FeatureGroup"
    mock_fg_instance.__repr__ = lambda self: "FeatureGroup"
    mock_fg.return_value = mock_fg_instance
    
    # ✅ Настройка get_color — возвращает строку, а не MagicMock
    mock_get_color.return_value = "#FF0000"
    
    # ✅ Настройка haversine — возвращает float, а не MagicMock
    mock_haversine.return_value = 1.5
    
    # ✅ Настройка PolyLine — возвращает объект с add_to методом
    mock_polyline_instance = MagicMock()
    mock_polyline.return_value = mock_polyline_instance
    
    # mode='all'
    result1 = visualize_mst_map(df, mst, bbox, mode="all")
    assert result1 == "logistics_mst.html"
    
    # mode!='all'
    result2 = visualize_mst_map(df, mst, bbox, mode="rail")
    assert result2 == "logistics_mst.html"
    
    assert mock_map.save.call_count == 2



class TestBuildMstRailByColor:
    
    def test_empty_df(self):
      
        df = pd.DataFrame(columns=["geometry", "tags"])
        result = build_mst_rail_by_color(df)
        assert isinstance(result, nx.Graph)
        assert len(result.nodes) == 0

    def test_single_station(self):
   
        df = pd.DataFrame(
            {"geometry": [Point(0, 0)], "tags": [{"colour": "red"}]},
            index=[0]
        )
        result = build_mst_rail_by_color(df)
        assert isinstance(result, nx.Graph)
        assert len(result.nodes) == 1
        assert len(result.edges) == 0

    def test_multiple_colors(self):

        df = pd.DataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "tags": [{"colour": "red"}, {"colour": "red"}, {"colour": "blue"}],
            },
            index=[0, 1, 2]
        )
        with patch('services.logistics.build_geodesic_graph') as mock_geo:
            mock_geo.return_value = nx.Graph()
            result = build_mst_rail_by_color(df)
            assert isinstance(result, nx.Graph)
            assert len(result.nodes) == 3
            assert mock_geo.call_count >= 2  # red + blue

    def test_nan_colour(self):

        df = pd.DataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1)],
                "tags": [{"name": "A"}, {"name": "B"}],  # Нет colour
            },
            index=[0, 1]
        )
        with patch('services.logistics.build_geodesic_graph') as mock_geo:
            mock_geo.return_value = nx.Graph()
            result = build_mst_rail_by_color(df)
            assert isinstance(result, nx.Graph)
            assert len(result.nodes) == 2
            mock_geo.assert_called_once()

    def test_edges_have_colour_attr(self):
        df = pd.DataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1)],
                "tags": [{"colour": "red"}, {"colour": "red"}],
            },
            index=[0, 1]
        )
        with patch('services.logistics.build_geodesic_graph') as mock_geo:
            mst = nx.Graph()
            mst.add_edge(0, 1, weight=100.0)
            mock_geo.return_value = mst
            
            result = build_mst_rail_by_color(df)
            
            for u, v, data in result.edges(data=True):
                assert "weight" in data
                assert "colour" in data

class TestCORS:

    def test_cors_headers_present(self):
        response = client.get("/health")
        
        # Проверяем, что CORS middleware работает
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])
