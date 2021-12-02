import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import ipywidgets as widgets
from ipywidgets import interact, fixed
from sklearn.neighbors import BallTree
import numpy as np


def execute_sql(conn, sql):
    """ Executes the SQL command supplied on the MariaDB database
       specified in the connection.

     Args:
        conn: connection to MariaDB database
        sql: sql command to be executed

    Returns:
        Pandas DataFrame containing the results of the SQL query with the same column
        names as those in the database
    """
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
    """ Returns the positions of interests specified in the tags and (optionally)
    value fields within the bounding box of the specified dimensions centered at
    the longitude/latitude provided

    Args:
       longitude: longitude at center of bounding box
       latitude: latitude at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       tags: dictionary of tags of features to be used in osmnx query
             e.g. {"amenity": True, "leisure": True}
       value: optional argument to specify more specific features
              e.g. tags = {"amenity": True}, value = "school"
              will return schools in the bounding box
    Returns:
       GeoDataFrame containing specified positions of interest
    """
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
    """ Returns the average house price in the bounding box specified for the given year

    Args:
       conn: connection to MariaDB database
       latitude: latitude at center of bounding box
       longitude: longitude at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       year: year in which to find average price
    Returns:
       DataFrame containing year and average house price
    """
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
    """ Returns the average house price in the bounding box specified for the given year

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       year: year in which to find average price
    Returns:
       DataFrame containing year and average house price
    """
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


def pois_in_area(conn, postcode, width, height, tags, value=None):
    """ Returns the positions of interests specified in the tags and (optionally)
        value fields within the bounding box of the specified dimensions centered at
        the longitude/latitude provided

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       tags: dictionary of tags of features to be used in osmnx query
             e.g. {"amenity": True, "leisure": True}
       value: optional argument to specify more specific features
              e.g. tags = {"amenity": True}, value = "school"
              will return schools in the bounding box
    Returns:
       GeoDataFrame containing specified positions of interest
    """
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


def plot_pois(conn, pois, postcode, width, height):
    """ Uses matplotlib to plot the positions of interest in the given area

    Args:
       conn: connection to MariaDB database
       pois: GeoDataFrame containing the positions of interest to be plotted
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
    """
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


def view_pois_interactive(conn, postcode, width, height, tags, year, value="None"):
    """ Creates a widget for visualisation of positions of interest in the given area

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       tags: dictionary of tags of features to be used in osmnx query
             e.g. {"amenity": True, "leisure": True}
       year: year to be used to calculate average house price in bounding box
       value: optional argument to specify more specific features
              e.g. tags = {"amenity": True}, value = "school"
              will return schools in the bounding box
    """
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
    """ Plots a graph of average house price vs year for each year in the
        time period specified

    Args:
       conn: connection to MariaDB database
       start_year: start year of time period (inclusive)
       end_year: end year of time period (inclusive)
    """
    year_prices = execute_sql(conn, f'SELECT year(date_of_transfer) AS year, AVG(price) FROM pp_data '
                                    f'WHERE year(date_of_transfer) BETWEEN {start_year} AND {end_year} GROUP BY year(date_of_transfer)')
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.plot(year_prices['year'], year_prices['AVG(price)'])
    plt.plot()
    plt.xlabel('year')
    plt.ylabel('average house price')


def house_price_vs_number_of_features(conn, postcode, width, height, distance_from_house, year, features_dict):
    """ Returns a DataFrame containing house price data as well as the number of given features within a bounding
        box around each house (see notebook for more detailed explanation)

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       distance_from_house: width/height in km of square around each house within which features should be counted
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around each house.
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format

    Returns:
       DataFrame containing house price data and the number of each of the given features within a bounding box
       around each house
    """
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


def house_price_vs_number_of_features_coordinates(conn, longitude, latitude, property_type, width, height,
                                                  distance_from_house, year, features_dict):
    """ Returns a DataFrame containing house price data as well as the number of given features within a bounding
        box around each house (see notebook for more detailed explanation)

    Args:
       conn: connection to MariaDB database
       longitude: longitude at center of bounding box
       latitude: latitude at center of bounding box
       property_type: type of property to select e.g. 'D' for detached houses
       width: width of bounding box in km
       height: height of bounding box in km
       distance_from_house: width/height in km of square around each house within which features should be counted
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around each house.
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format

    Returns:
       DataFrame containing house price data and the number of each of the given features within a bounding box
       around each house
    """
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


def get_test_features_without_distance(longitude, latitude, width, height, distance_from_house, year, features_dict):
    """ Given the longitude and latitude of a house from the model, this function will calculate the feature
    values required by the model (e.g. number of schools within a certain distance of the longitude/latitude
    specified). It will only calculate the number of the given features surrounding the house, and not the distances
    to the given features

    Args:
       longitude: longitude at center of bounding box
       latitude: latitude at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       distance_from_house: width/height in km of square around each house within which features should be counted
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around the house
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format

    Returns:
       DataFrame containing features required by the model to predict the price of a house
    """
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


def get_test_features_with_distance(longitude, latitude, width, height, distance_from_house, year, features_dict,
                                    distance_features_dict):
    """ Given the longitude and latitude of a house from the model, this function will calculate the feature
    values required by the model (e.g. number of schools within a certain distance of the longitude/latitude
    specified). It will calculate the number of the given features surrounding the house, and also the distances
    to the given features (e.g. distance to closest school)

    Args:
       longitude: longitude at center of bounding box
       latitude: latitude at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       distance_from_house: width/height in km of square around each house within which features should be counted
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around the house
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format
       distance_features_dict: dictionary which specifies which positions of interest to measure the distance to around
                               the house. Same format as "features_dict"

    Returns:
       DataFrame containing features required by the model to predict the price of a house
    """
    d = ((distance_from_house / 2) / 40075) * 360
    df = pd.DataFrame([[longitude, latitude]], columns=["longitude", "lattitude"])
    df["latitude_rad"] = np.deg2rad(df["lattitude"].values.astype(float))
    df["longitude_rad"] = np.deg2rad(df["longitude"].values.astype(float))

    for feature, tags in features_dict.items():
        if len(tags) == 0:
            features = pois_in_area_coordinates(longitude, latitude,
                                                width + 2 * distance_from_house,
                                                height + 2 * distance_from_house,
                                                {feature: True})
            features["latitude"] = features["geometry"].centroid.y
            features["longitude"] = features["geometry"].centroid.x
            num_features = len(
                features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                         & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
            df[f'{feature}_number'] = num_features
        else:
            for tag in tags:
                features = pois_in_area_coordinates(longitude, latitude,
                                                    width + 2 * distance_from_house,
                                                    height + 2 * distance_from_house,
                                                    {feature: True}, tag)
                features["latitude"] = features["geometry"].centroid.y
                features["longitude"] = features["geometry"].centroid.x
                num_features = len(
                    features[(features['longitude'] <= longitude + d) & (features['longitude'] >= longitude - d)
                             & (features['latitude'] <= latitude + d) & (features['latitude'] >= latitude - d)])
                df[f'{feature}_{tag}_number'] = num_features

    for feature, tags in distance_features_dict.items():
        if len(tags) == 0:
            pois = pois_in_area_coordinates(longitude, latitude, 2 * width, 2 * height, {
                feature: True})  # So houses in "corners" of houses box can find points outside box
            if len(pois) > 0:
                pois["latitude"] = pois["geometry"].centroid.y
                pois["longitude"] = pois["geometry"].centroid.x
                pois["latitude_rad"] = np.deg2rad(pois["latitude"].values)
                pois["longitude_rad"] = np.deg2rad(pois["longitude"].values)
                ball = BallTree(pois[["latitude_rad", "longitude_rad"]].values, metric='haversine')
                distances, indices = ball.query(df[["latitude_rad", "longitude_rad"]].values, k=1)
                df[f'distance_from_{feature}'] = distances * 6371
        else:
            for tag in tags:
                pois = pois_in_area_coordinates(longitude, latitude, 2 * width, 2 * height,
                                                {feature: True},
                                                tag)  # So houses in "corners" of houses box can find points outside box
                if len(pois) > 0:
                    pois["latitude"] = pois["geometry"].centroid.y
                    pois["longitude"] = pois["geometry"].centroid.x
                    pois["latitude_rad"] = np.deg2rad(pois["latitude"].values)
                    pois["longitude_rad"] = np.deg2rad(pois["longitude"].values)
                    ball = BallTree(pois[["latitude_rad", "longitude_rad"]].values, metric='haversine')
                    distances, indices = ball.query(df[["latitude_rad", "longitude_rad"]].values, k=1)
                    df[f'distance_from_{feature}_{tag}'] = distances * 6371
    return df


def house_price_vs_distance_from_feature_coordinates(conn, longitude, latitude, property_type, width, height, year,
                                                     features_dict):
    """ Returns a DataFrame containing house price data as well as the distance to the closest feature for each of
    the given features e.g. distance to closest school (see notebook for more detailed explanation)

    Args:
       conn: connection to MariaDB database
       longitude: longitude at center of bounding box
       latitude: latitude at center of bounding box
       property_type: type of property to select e.g. 'D' for detached houses
       width: width of bounding box in km
       height: height of bounding box in km
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around each house.
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format

    Returns:
           DataFrame containing house price data and the distance to the closest feature for each of the given features
    """
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


def house_price_vs_distance_from_feature(conn, postcode, width, height, year, features_dict):
    """ Returns a DataFrame containing house price data as well as the distance to the closest feature for each of
        the given features e.g. distance to closest school (see notebook for more detailed explanation)

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around each house.
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format

    Returns:
       DataFrame containing house price data and the distance to the closest feature for each of the given features
    """
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


def plot_house_price_vs_number_of_features(conn, postcode, width, height, distance_from_house, year, features_dict,
                                           max_price):
    """ Creates a plot of house price against the number of features surrounding each house for the parameters
    specified

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       distance_from_house: width/height in km of square around each house within which features should be counted
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around each house.
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format
       max_price: maximum house price to plot
    """
    data = house_price_vs_number_of_features(conn, postcode, width, height, distance_from_house, year, features_dict)
    data = data[data['price'] < max_price]
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


def plot_house_price_vs_distance_from_feature(conn, postcode, width, height, year, features_dict, max_price):
    """ Creates a plot of house price against the distance to the closest feature for each house for the parameters
        specified

    Args:
       conn: connection to MariaDB database
       postcode: postcode at center of bounding box
       width: width of bounding box in km
       height: height of bounding box in km
       year: year for which house price data should be selected
       features_dict: dictionary which specifies which positions of interest to count around each house.
                      e.g. {"amenity": ["cafe", "restaurant"], "leisure": ["park"]}
                      see notebook for more details about exact format
       max_price: maximum house price to plot
    """
    data = house_price_vs_distance_from_feature(conn, postcode, width, height, year, features_dict)
    data = data[data['price'] < max_price]
    plt.rcParams["figure.figsize"] = (10, 5)
    for feature, tags in features_dict.items():
        if len(tags) == 0:
            plt.figure()
            plt.scatter(data[f'distance_from_{feature}'], data['price'])
            plt.xlabel(f'Distance from closest {feature}')
            plt.ylabel('House Price')
            print(f'Correlation of distance from closest {feature} with price: ',
                  data[f'distance_from_{feature}'].corr(data['price']))
        else:
            for tag in tags:
                plt.figure()
                plt.rcParams["figure.figsize"] = (10, 5)
                plt.scatter(data[f'distance_from_{feature}_{tag}'], data['price'])
                plt.xlabel(f'Distance from closest {feature}_{tag}')
                plt.ylabel('House Price')
                print(f'Correlation of distance from closest {feature}_{tag} with price: ',
                      data[f'distance_from_{feature}_{tag}'].corr(data['price']))
    plt.show()


def interactive_viewer(conn):
    """ Creates a an interactive widget for viewing positions of interest using OpenStreetMaps

    Args:
       conn: connection to MariaDB database
    """
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
