import pymysql
import urllib.request


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
            quote = '"'
            upload_statement = (
                "LOAD DATA LOCAL INFILE %s INTO TABLE pp_data "
                "FIELDS TERMINATED BY ',' "
                f"OPTIONALLY ENCLOSED BY '{quote}' "  # One of these two should work!
                "LINES STARTING BY '' TERMINATED BY '\n';"
            )
            cur.execute(upload_statement, csv)
            print(f"Uploaded CSV: {csv}")
    conn.commit()
    conn.close()


def upload_postcode_data(conn):
    """Upload postcode data to the MariaDB database
       specified in the connection.

    Args:
        conn: Connection to MariaDB database

    """
    cur = conn.cursor()
    upload_statement = (
        "LOAD DATA LOCAL INFILE 'open_postcode_geo.csv' INTO TABLE postcode_data "
        "FIELDS TERMINATED BY ',' "
        "LINES STARTING BY '' TERMINATED BY '\n';"
    )
    cur.execute(upload_statement)
    conn.commit()
    print("Changes commmited")
    conn.close()
    print("Uploaded open_postcode_geo.csv")
