from .config import *
import pymysql
import urllib

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.

    Args:
        user: username
        password: password
        host: host url
        database: database
        port: port number

    Returns:
        Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print("MariaDB Server connection successful")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def upload_price_paid_data(conn, start_year, end_year):
    """ Retrieve the price paid data part by part and upload
        to the MariaDB database specified in the connection

    Args:
        conn: Connection to MariaDB database
        start_year: First year of data
        end_year: Final year of data

    """
    cur = conn.cursor()
    parts = ["part1", "part2"]
    for year in range(start_year, end_year + 1):
        for part in parts:
            url = f"http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}-{part}.csv"
            csv = f"price-paid-{year}-{part}.csv"
            urllib.request.urlretrieve(url, csv)
            cur.execute('''LOCAL DATA LOAD INFILE {} INTO TABLE {}
                           FIELDS TERMINATED BY ',' 
                           LINES STARTING BY '' TERMINATED BY '\\n';'''.format(csv, "pp_data"))
