from .config import *
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import mlai
import mlai.plot as plot
from . import access

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
