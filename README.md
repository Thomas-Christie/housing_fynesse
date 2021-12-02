# Housing Fynesse Repository

This is a python repo for using the Fynesse framework in order to analyse
house price data and predict house prices.

The code for this project is split into three main modules:
- access.py
- assess.py
- address.py

with the code in each module used to address each of the three aspects of the Fynesse framework.


## Access

The "access" module is used to upload the data to the database, in conjunction with some
inline SQL found in the notebook. It also provides the means by which a user can connect to the database to run
SQL queries elsewhere in the codebase. 

The data used for this project is the [UK Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
, which details the prices paid for houses from 1995 to the present. This is used in
conjunction with data from the UK Office for National Statistics on the longitude/latitude
of postcodes, which can be found [here](https://www.getthedata.com/open-postcode-geo).

It provides three functions to upload the data:
1. ```create_connection``` - Takes in a set of database credentials and returns a PyMySQL connection
object. This can then be used to easily execute SQL queries, and is used a lot throughout 
the codebase.
2. ```upload_price_paid_data``` - Takes in a connection object and uploads the UK Price Paid Data
to the database specified by the connection. Also take in a start year and end year for selecting what
range of data should be uploaded. The function automatically downloads the data for each year, which
is split into two parts, and uploads each part to the database.
3. ```upload_postcode_data``` - Takes in a connection object and uploads the postcode data to the
database specified by the connection. The postcode data CSV should be saved in a file called "open_postcode_geo.csv"
in the same directory as the notebook from which this function is being run.



## Assess

The "assess" module is used to assess the data provided. It provides means by which features
of the data can be analysed, so that the user can get a feel for the data being used. It also provides functions which
prepare the data and get it into a suitable format for the "address" module. Finally, it also provides
several functions to visualise the data. Some of these visualisations have been made with the task
of house price prediction in mind, so do straddle the boundary between the "assess" and "address" aspects of
the framework. I decided to place them in the "assess" module since they were more exploratory
in nature.

The functions provided are briefly described below. More detailed descriptions can be found in the module itself:

1. ```execute_sql``` - Executes a given SQL query and returns the result in a dataframe.
2. ```pois_in_area``` - Returns a GeoDataFrame from OpenStreetMaps containing the points of interest 
in a given area. The specific types of points can also be specified (e.g. "parks")
3. ```pois_in_area_coordinates``` - As above, but uses longitude/latitude to specify an area rather than a postcode.
4. ```price_in_box_in_year``` - Calculates the average price of houses in a bounding box
specified for a given year.
5. ```price_in_box_in_year_coordinates``` - As above, but uses longitude/latitude to specify an area rather than a postcode.
6. ```plot_pois``` - Used to produce a plot points of interest, so that the user can view them
on a map for easy visualisation.
7. ```view_pois_interactive``` - Used to create an interactive widget through which the user can  visualise
different points of interest in different areas.
8. ```year_avg_house_price``` - Calculates the average house price for each year in the range specified.
9. ```house_price_vs_number_of_features``` - Takes some parameters which are used to create a bounding
box and calculates features for houses within the box. The features calculated by this
function are the numbers of given points surrounding each house (e.g. number of schools within a 
1km box around each house). The data is returned in the form of a dataframe, and is used for training the models
in the "address" stage of the framework.
10. ```house_price_vs_number_of_features_coordinates``` - As above, but uses longitude/latitude to specify an area 
rather than a postcode.
11. ```get_test_feature_numbers``` - Takes in a latitude and longitude and prepares a dataframe which can be input
to the model to get a predicted price for a given location. It calculates features concerned
with the number of points of interest surrounding a house (e.g. number of schools within a 
1km box of the input longitude/latitude).
12. ```get_test_feature_distances``` - Takes in a latitude and longitude and prepares a dataframe which can be
used as input to a model to get a predicted price for a given location. It calculates features 
concerned with the distance to surrounding points of interest (e.g. distance to closest school).
13. ```house_price_vs_distance_from_feature``` -  Takes some parameters which are used to create a bounding
box and calculates features for houses within the box. The features calculated by this
function are the distances to given points for each house (e.g. distance to closest school for each house). The data is
returned in the form of a dataframe, and is used for training the models in the "address" stage of the framework.
14. ```house_price_vs_distance_from_feature_coordinates``` - As above, but uses longitude/latitude to specify an area rather than a postcode.
15. ```plot_house_price_vs_number_of_features``` - Used to plot house prices against numbers of features for user-specified features.
This can be used to help answer questions such as "How does the number of schools close to a house affect its price?"
16. ```plot_house_price_vs_distance_from_feature``` - Used to plot house prices against distance from features for user-specified
features. This can be used to help answer questions such as "How does the distance to the closes cafe affect the price of a house?"
17. ```interactive_viewer```- Used to display the interactive widget for displaying points of interest.

## Address

The "address" module is used to predict house prices, and contains the functions for my models.
I have two models - one which takes into account only the *numbers* of features surrounding houses
and one which also takes into account the *distances* to features surrounding houses. The model which takes
into account distances tends to outperform the other model, but there are occasionally areas when this 
is reversed, which is why I left both models in for comparison.

There are two functions which define the two models:

1. ```predict_price_without_distance```
2. ```predict_price_with_distance```

They take as input a longitude, latitude, year and property type (as well as a connection object for the database)
and predict a house price. More details of how the models work can be found in the notebook.