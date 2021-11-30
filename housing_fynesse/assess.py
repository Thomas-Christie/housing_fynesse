from .config import *
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import mlai
import mlai.plot as plot
import ipywidgets as widgets
from ipywidgets import interact, fixed
from . import access
from sklearn.neighbors import BallTree, KDTree
import numpy as np

"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def execute_sql(conn, sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        field_names = [i[0] for i in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=field_names)
        return df
    except Exception as e:
        print(e)
        cur.close()


def pois_in_area_coordinates(longitude, latitude, width, height, tags, value=None):
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    if value is not None and len(pois) > 0:
        pois = pois[pois[list(tags.keys())[0]] == value]
    return pois


def price_in_box_in_year_coordinates(conn, latitude, longitude, width, height, year):
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    price = execute_sql(conn, f'SELECT year(pp.date_of_transfer) AS year, AVG(pp.price) as average_price FROM '
                              f'(SELECT * FROM postcode_data WHERE lattitude <= {north} AND lattitude >= {south} AND longitude <= {east} AND longitude >= {west}) AS post '
                              f'INNER JOIN '
                              f'(SELECT * FROM pp_data WHERE year(date_of_transfer) = {year}) as pp '
                              f'ON '
                              f'pp.postcode = post.postcode')
    return price


def price_in_box_in_year_postcode(conn, postcode, width, height, year):
    longitude = float(
        execute_sql(conn, f"SELECT longitude FROM postcode_data WHERE postcode='{postcode}'")['longitude'])
    latitude = float(execute_sql(conn, f"SELECT lattitude FROM postcode_data WHERE postcode='{postcode}'")['lattitude'])
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    price = execute_sql(conn, f'SELECT year(pp.date_of_transfer) AS year, AVG(pp.price) as average_price FROM '
                              f'(SELECT * FROM postcode_data WHERE lattitude <= {north} AND lattitude >= {south} AND longitude <= {east} AND longitude >= {west}) AS post '
                              f'INNER JOIN '
                              f'(SELECT * FROM pp_data WHERE year(date_of_transfer) = {year}) as pp '
                              f'ON '
                              f'pp.postcode = post.postcode')
    return price


# Added conn
def pois_in_area(conn, postcode, width, height, tags, value=None):
    longitude = float(
        execute_sql(conn, f"SELECT longitude FROM postcode_data WHERE postcode='{postcode}'")['longitude'])
    latitude = float(execute_sql(conn, f"SELECT lattitude FROM postcode_data WHERE postcode='{postcode}'")['lattitude'])
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    if value is not None:
        pois = pois[pois[list(tags.keys())[0]] == value]
    print(
        f"There are {len(pois)} points of interest with tag={tags} taking value={value} surrounding {postcode} latitude: {latitude}, longitude: {longitude}")
    return pois


# Added conn
def plot_pois(conn, pois, postcode, width, height):
    longitude = float(
        execute_sql(conn, f"SELECT longitude FROM postcode_data WHERE postcode='{postcode}'")['longitude'])
    latitude = float(execute_sql(conn, f"SELECT lattitude FROM postcode_data WHERE postcode='{postcode}'")['lattitude'])
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    fig, ax = plt.subplots(figsize=[10, 10])
    graph = ox.graph_from_bbox(north, south, east, west)
    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)
    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # Plot all POIs
    if pois.empty:
        print("No pois to show")
    else:
        pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.show()


# Added conn
def view_pois_interactive(conn, postcode, width, height, tags, year, value="None"):
    tags_dict = {}
    for tag in tags:
        tags_dict[tag] = True
    if value.lower() == 'none':
        p = pois_in_area(conn, postcode, width, height, tags_dict)
    else:
        p = pois_in_area(conn, postcode, width, height, tags_dict, value)
    avg_house_price = price_in_box_in_year_postcode(conn, postcode, width, height, year)
    print(
        f"Average house price in box of width = {width}km, height = {height}km, center = {postcode} in year {year} is: £{avg_house_price['average_price'][0]}")
    plot_pois(conn, p, postcode, width, height)


def year_avg_house_price(conn, start_year, end_year):
    year_prices = execute_sql(conn, f'SELECT year(date_of_transfer) AS year, AVG(price) FROM pp_data '
                                    f'WHERE year(date_of_transfer) BETWEEN {start_year} AND {end_year} GROUP BY year(date_of_transfer)')
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.plot(year_prices['year'], year_prices['AVG(price)'])
    plt.plot()
    plt.xlabel('year')
    plt.ylabel('average house price')


# Added conn
def house_price_vs_number_of_features(conn, postcode, width, height, distance_from_house, year, features_dict):
    longitude = float(
        execute_sql(conn, f"SELECT longitude FROM postcode_data WHERE postcode='{postcode}'")['longitude'])
    latitude = float(execute_sql(conn, f"SELECT lattitude FROM postcode_data WHERE postcode='{postcode}'")['lattitude'])
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    d = ((distance_from_house / 2) / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    houses = execute_sql(conn, f'SELECT * FROM '
                               f'(SELECT * FROM postcode_data WHERE lattitude <= {north} AND lattitude >= {south} AND longitude <= {east} AND longitude >= {west}) AS post '
                               f'INNER JOIN '
                               f'(SELECT * FROM pp_data WHERE year(date_of_transfer) = {year}) as pp '
                               f'ON '
                               f'pp.postcode = post.postcode')
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            features = pois_in_area(conn, postcode, width + 2 * distance_from_house, height + 2 * distance_from_house,
                                    {feature: True})
            features["latitude"] = features["geometry"].centroid.y
            features["longitude"] = features["geometry"].centroid.x
            a = []
            postcode_features = {}
            for _, row in houses.iterrows():
                longitude = float(row['longitude'])
                latitude = float(row['lattitude'])
                if row['postcode'][0] in postcode_features:
                    a.append(postcode_features[row['postcode'][0]])
                else:
                    num_features = len(
                        features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                                 & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
                    postcode_features[row['postcode'][0]] = num_features
                    a.append(num_features)
            houses[f'{feature}_number'] = a
        else:
            for tag in tags:
                features = pois_in_area(conn, postcode, width + 2 * distance_from_house,
                                        height + 2 * distance_from_house,
                                        {feature: True}, tag)
                features["latitude"] = features["geometry"].centroid.y
                features["longitude"] = features["geometry"].centroid.x
                a = []
                postcode_features = {}
                for _, row in houses.iterrows():
                    longitude = float(row['longitude'])
                    latitude = float(row['lattitude'])
                    if row['postcode'][0] in postcode_features:
                        a.append(postcode_features[row['postcode'][0]])
                    else:
                        num_features = len(
                            features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                                     & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
                        postcode_features[row['postcode'][0]] = num_features
                        a.append(num_features)
                houses[f'{feature}_{tag}_number'] = a
    return houses


# Added conn
def house_price_vs_number_of_features_coordinates(conn, longitude, latitude, property_type, width, height,
                                                  distance_from_house, year, features_dict):
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    d = ((distance_from_house / 2) / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    houses = execute_sql(conn, f'SELECT * FROM '
                               f'(SELECT * FROM postcode_data WHERE lattitude <= {north} AND lattitude >= {south} AND longitude <= {east} AND longitude >= {west}) AS post '
                               f'INNER JOIN '
                               f'(SELECT * FROM pp_data WHERE year(date_of_transfer) BETWEEN {year - 1} AND {year + 1} AND property_type = "{property_type}") as pp '
                               f'ON '
                               f'pp.postcode = post.postcode')
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            features = pois_in_area_coordinates(longitude, latitude, width + 2 * distance_from_house,
                                                height + 2 * distance_from_house, {feature: True})
            features["latitude"] = features["geometry"].centroid.y
            features["longitude"] = features["geometry"].centroid.x
            a = []
            postcode_features = {}
            for _, row in houses.iterrows():
                longitude = float(row['longitude'])
                latitude = float(row['lattitude'])
                if row['postcode'][0] in postcode_features:
                    a.append(postcode_features[row['postcode'][0]])
                else:
                    num_features = len(
                        features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                                 & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
                    postcode_features[row['postcode'][0]] = num_features
                    a.append(num_features)
            houses[f'{feature}_number'] = a
        else:
            for tag in tags:
                features = pois_in_area_coordinates(longitude, latitude, width + 2 * distance_from_house,
                                                    height + 2 * distance_from_house, {feature: True}, tag)
                features["latitude"] = features["geometry"].centroid.y
                features["longitude"] = features["geometry"].centroid.x
                a = []
                postcode_features = {}
                for _, row in houses.iterrows():
                    longitude = float(row['longitude'])
                    latitude = float(row['lattitude'])
                    if row['postcode'][0] in postcode_features:
                        a.append(postcode_features[row['postcode'][0]])
                    else:
                        num_features = len(
                            features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                                     & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
                        postcode_features[row['postcode'][0]] = num_features
                        a.append(num_features)
                houses[f'{feature}_{tag}_number'] = a
    return houses


def number_of_features_surrounding_test(longitude, latitude, width, height, distance_from_house, year, features_dict):
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    d = ((distance_from_house / 2) / 40075) * 360
    df = pd.DataFrame([[longitude, latitude]], columns=["longitude", "lattitude"])
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            features = pois_in_area_coordinates(longitude, latitude, width + 2 * distance_from_house,
                                                height + 2 * distance_from_house, {feature: True})
            features["latitude"] = features["geometry"].centroid.y
            features["longitude"] = features["geometry"].centroid.x
            num_features = len(
                features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                         & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
            df[f'{feature}_number'] = num_features
        else:
            for tag in tags:
                features = pois_in_area_coordinates(longitude, latitude, width + 2 * distance_from_house,
                                                    height + 2 * distance_from_house, {feature: True}, tag)
                features["latitude"] = features["geometry"].centroid.y
                features["longitude"] = features["geometry"].centroid.x
                num_features = len(
                    features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                             & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
                df[f'{feature}_{tag}_number'] = num_features
    return df


# Added conn
def find_nearest_point_coordinates(conn, longitude, latitude, property_type, width, height, year, features_dict):
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    houses = execute_sql(conn, f'SELECT * FROM '
                               f'(SELECT * FROM postcode_data WHERE lattitude <= {north} AND lattitude >= {south} AND longitude <= {east} AND longitude >= {west}) AS post '
                               f'INNER JOIN '
                               f'(SELECT * FROM pp_data WHERE year(date_of_transfer) BETWEEN {year - 1} AND {year + 1}) as pp '
                               f'ON '
                               f'pp.postcode = post.postcode')
    print("Number of houses: ", len(houses))
    houses["latitude_rad"] = np.deg2rad(houses["lattitude"].values.astype(float))
    houses["longitude_rad"] = np.deg2rad(houses["longitude"].values.astype(float))
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            pois = pois_in_area_coordinates(longitude, latitude, 2 * width, 2 * height, {
                feature: True})  # So houses in "corners" of houses box can find points outside box
            if len(pois) > 0:
                pois["latitude"] = pois["geometry"].centroid.y
                pois["longitude"] = pois["geometry"].centroid.x
                pois["latitude_rad"] = np.deg2rad(pois["latitude"].values)
                pois["longitude_rad"] = np.deg2rad(pois["longitude"].values)
                ball = BallTree(pois[["latitude_rad", "longitude_rad"]].values, metric='haversine')
                distances, indices = ball.query(houses[["latitude_rad", "longitude_rad"]].values, k=1)
                houses[f'distance_from_{feature}'] = distances * 6371
        else:
            for tag in tags:
                pois = pois_in_area_coordinates(longitude, latitude, 2 * width, 2 * height, {feature: True},
                                                tag)  # So houses in "corners" of houses box can find points outside box
                if len(pois) > 0:
                    pois["latitude"] = pois["geometry"].centroid.y
                    pois["longitude"] = pois["geometry"].centroid.x
                    pois["latitude_rad"] = np.deg2rad(pois["latitude"].values)
                    pois["longitude_rad"] = np.deg2rad(pois["longitude"].values)
                    ball = BallTree(pois[["latitude_rad", "longitude_rad"]].values, metric='haversine')
                    distances, indices = ball.query(houses[["latitude_rad", "longitude_rad"]].values, k=1)
                    houses[f'distance_from_{feature}_{tag}'] = distances * 6371
    return houses


# Added conn
def find_nearest_point(conn, postcode, width, height, year, features_dict):
    longitude = float(
        execute_sql(conn, f"SELECT longitude FROM postcode_data WHERE postcode='{postcode}'")['longitude'])
    latitude = float(execute_sql(conn, f"SELECT lattitude FROM postcode_data WHERE postcode='{postcode}'")['lattitude'])
    box_width = (width / 40075) * 360
    box_height = (height / 40075) * 360
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2
    houses = execute_sql(conn, f'SELECT * FROM '
                               f'(SELECT * FROM postcode_data WHERE lattitude <= {north} AND lattitude >= {south} AND longitude <= {east} AND longitude >= {west}) AS post '
                               f'INNER JOIN '
                               f'(SELECT * FROM pp_data WHERE year(date_of_transfer) = {year}) as pp '
                               f'ON '
                               f'pp.postcode = post.postcode')
    houses["latitude_rad"] = np.deg2rad(houses["lattitude"].values.astype(float))
    houses["longitude_rad"] = np.deg2rad(houses["longitude"].values.astype(float))
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            pois = pois_in_area(conn, postcode, 2 * width, 2 * height,
                                {feature: True})  # So houses in "corners" of houses box can find points outside box
            pois["latitude"] = pois["geometry"].centroid.y
            pois["longitude"] = pois["geometry"].centroid.x
            pois["latitude_rad"] = np.deg2rad(pois["latitude"].values)
            pois["longitude_rad"] = np.deg2rad(pois["longitude"].values)
            ball = BallTree(pois[["latitude_rad", "longitude_rad"]].values, metric='haversine')
            distances, indices = ball.query(houses[["latitude_rad", "longitude_rad"]].values, k=1)
            houses[f'distance_from_{feature}'] = distances * 6371
        else:
            for tag in tags:
                pois = pois_in_area(conn, postcode, 2 * width, 2 * height, {feature: True},
                                    tag)  # So houses in "corners" of houses box can find points outside box
                pois["latitude"] = pois["geometry"].centroid.y
                pois["longitude"] = pois["geometry"].centroid.x
                pois["latitude_rad"] = np.deg2rad(pois["latitude"].values)
                pois["longitude_rad"] = np.deg2rad(pois["longitude"].values)
                ball = BallTree(pois[["latitude_rad", "longitude_rad"]].values, metric='haversine')
                distances, indices = ball.query(houses[["latitude_rad", "longitude_rad"]].values, k=1)
                houses[f'distance_from_{feature}_{tag}'] = distances * 6371
    return houses


def plot_house_price_vs_number_of_features(conn, postcode, width, height, distance_from_house, year, features_dict):
    data = house_price_vs_number_of_features(conn, postcode, width, height, distance_from_house, year, features_dict)
    data = data[data['price'] < 1000000]
    plt.rcParams["figure.figsize"] = (10, 5)
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            plt.figure()
            plt.scatter(data[f'{feature}_number'], data['price'])
            plt.xlabel(f'{feature}_number within {distance_from_house}km from house')
            plt.ylabel('House Price')
            print(f'Correlation of {feature}_number with price: ', data[f'{feature}_number'].corr(data['price']))
        else:
            for tag in tags:
                plt.figure()
                plt.rcParams["figure.figsize"] = (10, 5)
                plt.scatter(data[f'{feature}_{tag}_number'], data['price'])
                plt.xlabel(f'{feature}_{tag}_number within {distance_from_house}km from house')
                plt.ylabel('House Price')
                print(f'Correlation of {feature}_{tag}_number with price: ',
                      data[f'{feature}_{tag}_number'].corr(data['price']))
    plt.show()


def interactive_viewer(conn):
    postcode_select = widgets.Text(value='CB2 1RF', placeholder='CB2 1RF', description='Postcode:', disabled=False)
    # Select multiple by holding down shift
    tags_select = widgets.SelectMultiple(
        options=['amenity', 'buildings', 'historic', 'leisure', 'shop', 'tourism', 'public_transport'],
        value=['amenity'], description='Tags')
    value_select = widgets.Text(value='None', placeholder='None', description='Tag Value:')
    width_value = widgets.BoundedFloatText(value=1, min=0, max=20, step=0.1, description='Width:')
    height_value = widgets.BoundedFloatText(value=1, min=0, max=20, step=0.1, description='Height:')
    year_value = widgets.IntSlider(value=2010, min=1995, max=2021, step=1, description='Year:')
    return interact(view_pois_interactive,
                    conn=fixed(conn),
                    postcode=postcode_select,
                    width=width_value,
                    height=height_value,
                    tags=tags_select,
                    year=year_value,
                    value=value_select)


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
