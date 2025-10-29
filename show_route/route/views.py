import folium
from django.shortcuts import render, redirect


def showmap(request):
    return render(request,'showmap.html')

import osmnx as ox
import numpy as np
import os
import pandas as pd
import geopandas as gd
import networkx as nx
import geopy
from geopy.geocoders import Nominatim
from shapely import geometry, wkt
from scgraph.geographs.marnet import marnet_geograph


ox.settings.bidirectional_network_types = ["drive"]

''' Определение тегов '''
tags_port = {'industrial' : 'port'}
tags_aeroway = {'aeroway' : ['aerodrome','heliport', 'airstrip']}
tags_landuse = {'landuse' : 'railway'}
# tags_build = {'building' : 'warehouse', 'amenity' : 'mailroom'}

''' Построение датафрейма нужных фич '''
def func_tags(m_tags, point_y, point_x):

    try:
        gdf = ox.features.features_from_bbox(bbox = (max(point_y), min(point_y), min(point_x), max(point_x)), tags = m_tags).reset_index()
        gdf = gdf[['element_type', 'osmid']]
        gdf['n_osmid'] = gdf['element_type'].apply(lambda x: x[0]) + gdf['osmid'].astype(str)
        list_lat_lon = []
        for i in range(gdf.n_osmid.shape[0]):
            try:
                bb = ox.geocode_to_gdf(gdf.n_osmid.iloc[i], by_osmid=True)[['lat', 'lon']]
                list_lat_lon.append(bb.iloc[:])
            except ox._errors.InsufficientResponseError:
                list_lat_lon.append(pd.DataFrame({'lat': [None], 'lon': [None]}))
            except KeyError:
                list_lat_lon.append(pd.DataFrame({'lat': [None], 'lon': [None]}))
        dfs = pd.concat(list_lat_lon, axis=0).reset_index(drop=True)
        gdf = gdf.join(dfs, how='inner')
        gdf = gdf[['lat', 'lon']]
        gdf['kind_of'] = list(m_tags.keys())[0]
        gdf = gdf.dropna().reset_index(drop=True)
        return gdf
    except ox._errors.InsufficientResponseError:
        return None
    

''' Построение графа, получение датафрейма фич и получение датафрейма долгот и широт для дальшейших вычислений '''
def create_graph_city(point_y, point_x, my_network_type = None, my_filter = None):
    # full_df = pd.concat([func_tags(tags_port, point_y, point_x), 
    #                                 func_tags(tags_aeroway, point_y, point_x), 
    #                                 func_tags(tags_landuse, point_y, point_x), 
    #                                 func_tags(tags_build, point_y, point_x)], ignore_index=True)
    full_df = pd.concat([func_tags(tags_port, point_y, point_x), 
                                    func_tags(tags_aeroway, point_y, point_x), 
                                    func_tags(tags_landuse, point_y, point_x)], ignore_index=True)
    G = ox.graph_from_bbox(bbox = (max(point_y), min(point_y), min(point_x), max(point_x)), 
                           retain_all=True, simplify = True, network_type = my_network_type, custom_filter = my_filter)

    lat = list(full_df['lat'].values)
    lon = list(full_df['lon'].values)

    return full_df, G, lat, lon


''' Формирование датафрейма фич '''
def create_features_city(features_df, city_graph, lat, lon):
    features_df['new_nodes'] = ox.distance.nearest_nodes(city_graph, lon, lat)
    return features_df


''' Создание датафрейма со всеми маршрутами '''
def create_graph_route(input_graph, f_df):
    one_route = []
    for i in range(f_df.shape[0]):
        for j in range(i + 1, f_df.shape[0]):
            try:
                one_route.append(nx.shortest_path(input_graph, f_df['new_nodes'].iloc[i], f_df['new_nodes'].iloc[j]))
            except nx.NetworkXNoPath:
                pass
    route_df = pd.DataFrame({'route' : one_route})
    return route_df


''' Построение всех морских маршрутов '''
def create_route_marine(G, f_df):
    feat_port = f_df[f_df['kind_of'] == 'industrial']

    full_df_marine_x = list(feat_port.lon)
    full_df_marine_y = list(feat_port.lat)

    list_lat_marine = []
    list_lon_marine = []

    for i in range(len(full_df_marine_x)):
        for j in range(i + 1, len(full_df_marine_y)):
            output = (marnet_geograph.get_shortest_path(
                origin_node={"latitude": full_df_marine_y[i],"longitude": full_df_marine_x[i]}, 
                destination_node={"latitude": full_df_marine_y[j],"longitude": full_df_marine_x[j]}
            ))
            list_lat_marine.append([output['coordinate_path'][i][0] for i in range(len(output['coordinate_path']))])
            list_lon_marine.append([output['coordinate_path'][i][1] for i in range(len(output['coordinate_path']))])

    nodes = []
    list_nodes = []

    for i in range(len(list_lat_marine)):
        for j in range(len(list_lat_marine[i])):
            nodes.append(ox.distance.nearest_nodes(G, list_lon_marine[i][j], list_lat_marine[i][j]))
        list_nodes.append(nodes)
        nodes = []

    route_df = pd.DataFrame({'route' : list_nodes})
    return route_df


''' Построение всех аэромаршрутов '''
def create_route_aero(G, f_df):
    feat_aero = f_df[f_df['kind_of'] == 'aeroway']

    list_nodes = []

    for i in range(len(feat_aero.lat.values) - 1):
        for j in range(i + 1, len(feat_aero.lat.values)):
            nodes_1 = ox.distance.nearest_nodes(G, feat_aero.lon.iloc[i], feat_aero.lat.iloc[i])
            nodes_2 = ox.distance.nearest_nodes(G, feat_aero.lon.iloc[j], feat_aero.lat.iloc[j])
            list_nodes.append([nodes_1, nodes_2])

    route_df = pd.DataFrame({'route' : list_nodes})
    return route_df


''' Создание из нескольких LineString один MultiLineString - нужно для финального графа '''
def create_gdf_graph(nn, ee):
    ii = 0
    list_line = []
    list_multi_line = []
    for i in range(nn.shape[0] - 1):
        idx = nn.iloc[i + 1].new_nodes
        for j in range(ii, ee.index.get_level_values('u').shape[0]):
            if ee.index.get_level_values('u')[j] != idx:
                for k in range(len(ee.loc[ee.index.get_level_values('u')[j]]['geometry'].values)):
                    list_line.append(ee.loc[ee.index.get_level_values('u')[j]]['geometry'].values[k])
            else : 
                ii = j
        multi_line = geometry.MultiLineString(list_line)
        list_line = []
        list_multi_line.append(multi_line)
    return list_multi_line


''' Построение финального графа путей '''
def create_final_graph(arb_graph, route_df, feature_df):
    list_graphs = []
    for i in range(route_df.route.shape[0]):
        r = route_df['route'].values[i]

        rr = pd.DataFrame({'new_nodes' : r})
        rr['osmid'] = rr['new_nodes']
        rr = rr.set_index('osmid')
        rr['x'] = rr.new_nodes.apply(lambda x: arb_graph.nodes[x]['x'])
        rr['y'] = rr.new_nodes.apply(lambda x: arb_graph.nodes[x]['y'])
        rr = rr.drop_duplicates(subset=['new_nodes'])

        edge_dict = {'u': rr.new_nodes[:-1].values, 'v': rr.new_nodes[1:].values, 'key': 0}
        edge_gdf = gd.GeoDataFrame(edge_dict, crs=None)
        edge_gdf = edge_gdf.set_index(['u', 'v', 'key'])

        graph_attrs = {"crs": "WGS84"}
        multi_digraph_aero = ox.convert.graph_from_gdfs(
            rr, edge_gdf, graph_attrs=graph_attrs)
        
        list_graphs.append(multi_digraph_aero)
        
    my_graph = list_graphs[0]
    for i in range(1, len(list_graphs)):
        my_graph = nx.compose_all([my_graph, list_graphs[i]])
    
    nodes, edges = ox.graph_to_gdfs(my_graph)
    list_nodes = list(feature_df['new_nodes'].values)
    n = nodes[nodes['new_nodes'].isin(list_nodes)]

    list_multiline = create_gdf_graph(n, edges)
    edge_dict = {'u': n.new_nodes[:-1].values, 'v': n.new_nodes[1:].values, 'key': 0}
    edge_gdf = gd.GeoDataFrame(edge_dict, crs=None)
    edge_gdf = edge_gdf.set_index(['u', 'v', 'key'])
    edge_gdf['geometry'] = list_multiline
    graph_attrs = {"crs": "WGS84"}
    multi_digraph = ox.graph_from_gdfs(
        n, edge_gdf, graph_attrs=graph_attrs)
    
    return multi_digraph



''' Построение результирующего графа для автомобилей '''
def result(point_y, point_x):
    graph_city = create_graph_city(point_y, point_x, my_network_type = 'drive')
    feature_df = create_features_city(*graph_city)
    ox.distance.add_edge_lengths(graph_city[1], edges=None)
    route_df = create_graph_route(graph_city[1], feature_df)
    final_graph_1_drive = create_final_graph(graph_city[1], route_df, feature_df)
    return final_graph_1_drive, route_df

''' Построение результирующего графа для морей '''
def result_marine(point_y, point_x):
    graph_city = create_graph_city(point_y, point_x, my_network_type = 'drive')
    feature_df = create_features_city(*graph_city)
    ox.distance.add_edge_lengths(graph_city[1], edges=None)
    route_df = create_route_marine(graph_city[1], feature_df)
    try:
    	final_graph_1_marine = create_final_graph(graph_city[1], route_df, feature_df)
    except IndexError:
    	final_graph_1_marine = nx.MultiDiGraph()
    return final_graph_1_marine, route_df

''' Построение результирующего графа для воздушных путей '''
def result_aero(point_y, point_x):
    graph_city = create_graph_city(point_y, point_x, my_network_type = 'drive')
    feature_df = create_features_city(*graph_city)
    ox.distance.add_edge_lengths(graph_city[1], edges=None)
    route_df = create_route_aero(graph_city[1], feature_df)
    try:
    	final_graph_1_aero = create_final_graph(graph_city[1], route_df, feature_df)
    except IndexError:
    	final_graph_1_aero = nx.MultiDiGraph()
    return final_graph_1_aero, route_df


def showroutes(request, metrics, lat1, long1, lat2, long2):
    #locator = Nominatim(user_agent = "myapp")
    #loc = locator.reverse(lat1 + ' , ' + long1)
    #address = loc.raw['address']
    #city_from = address.get('city', '')

    figure = folium.Figure()
    m = folium.Map(location=[((float(lat1)+float(lat2))/2),
                                 ((float(long1)+float(long2))/2)], 
                       zoom_start=10)
    m.add_to(figure)

    point_y = (float(lat1), float(lat2))
    point_x = (float(long1), float(long2))

    if(os.path.exists('../Analytics/nodes_' + lat1 + ',' + long1 + ',' + lat2 + ',' + long2 + '.csv') and 
       os.path.exists('../Analytics/edges_' + lat1 + ',' + long1 + ',' + lat2 + ',' + long2 + '.csv')):
        gdf_nodes = pd.read_csv('../Analytics/nodes_' + lat1 + ',' + long1 + ',' + lat2 + ',' + long2 + '.csv', delimiter=",")
        gdf_edges = pd.read_csv('../Analytics/edges_' + lat1 + ',' + long1 + ',' + lat2 + ',' + long2 + '.csv', delimiter=",")

        gdf_nodes_new = gdf_nodes.set_index('osmid').drop(['new_nodes'], axis=1)
        gdf_nodes_new = gdf_nodes_new[['y', 'x', 'geometry']]
        gdf_edges_new = gdf_edges.set_index(['u', 'v', 'key'])

        gdf_edges_new['geometry'] = gdf_edges_new['geometry'].apply(wkt.loads)
        gdf_nodes_new['geometry'] = gdf_nodes_new['geometry'].apply(wkt.loads)

        df_edges = gd.GeoDataFrame(gdf_edges_new, crs = 'WGS84')
        df_nodes = gd.GeoDataFrame(gdf_nodes_new, crs = 'WGS84')

        graph = ox.graph_from_gdfs(df_nodes, df_edges)
    else:

        res_drive, drive_route_df = result(point_y, point_x)
        res_marine, marine_route_df = result_marine(point_y, point_x)
        res_aero, aero_route_df = result_aero(point_y, point_x)
        res_railway = ox.graph_from_bbox(bbox = (max(point_y), min(point_y), min(point_x), max(point_x)), 
                           retain_all=True, simplify = True, custom_filter = '["railway"~"rail"]')
    
        graph = nx.compose_all([res_drive, res_aero, res_railway, res_marine])

        df_nodes, df_edges = ox.graph_to_gdfs(graph)
        df_nodes.to_csv('../Analytics/nodes_' + lat1 + ',' + long1 + ',' + lat2 + ',' + long2 + '.csv')
        df_edges.to_csv('../Analytics/edges_' + lat1 + ',' + long1 + ',' + lat2 + ',' + long2 + '.csv')





    metrics_1 = nx.degree_centrality(graph)
    metrics_2 = nx.closeness_centrality(graph)
    metrics_3 = nx.betweenness_centrality(graph)
    metrics_4 = nx.pagerank(graph, alpha = 0.8)
    
    folium.GeoJson(df_edges).add_to(m)

    chosen_metrics = metrics

    match chosen_metrics:
        case "degree":
            df_nodes["metric"] = df_nodes.index.map(metrics_1)
        case "closeness":
            df_nodes["metric"] = df_nodes.index.map(metrics_2)
        case "betweenness":
            df_nodes["metric"] = df_nodes.index.map(metrics_3)
        case "pagerank":
            df_nodes["metric"] = df_nodes.index.map(metrics_4)
        case _:
            df_nodes["metric"] = 0

    df_nodes["marker_color"] = pd.cut(df_nodes["metric"], bins=4, labels=['green', 'yellow', 'orange', 'red'])
    for index, poi in df_nodes.iterrows():
        folium.features.CircleMarker(
            [df_nodes.loc[index, 'y'], df_nodes.loc[index, 'x']], radius = 5,
            color = df_nodes.loc[index, 'marker_color'], fill_color = df_nodes.loc[index, 'marker_color'], 
            fill = True, fill_opacity=0.7, popup=folium.Popup(str(df_nodes.loc[index, 'metric']))
        ).add_to(m)
    #folium.GeoJson(gdf_nodes).add_to(m)

    figure.render()
    context={'map':figure}
    return render(request,'showroute.html',context)
