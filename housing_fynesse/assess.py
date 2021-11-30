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


def year_avg_house_price(start_year, end_year):
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
