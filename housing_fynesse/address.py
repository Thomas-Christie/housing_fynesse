# This file contains code for supporting addressing questions in the data
import housing_fynesse.assess as assess
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def predict_price_without_distance(conn, latitude, longitude, year, property_type):
    """ Trains a model which then produces a price prediction for a house taking the parameters given.
    Model only takes into account the number of each feature surrounding the house.
    See notebook for more specific details about the model.

    Args:
       conn: connection to MariaDB database
       latitude: latitude of house
       longitude: longitude of house
       year: year for which house price should be predicted
       property_type: type of property e.g. 'D' for detached house

    """
    features = {"amenity": ["cafe", "restaurant", "school", "college", "bar"],
                "public_transport": [],
                "leisure": ["park"]}
    box_width = 4
    box_height = 4
    distance_from_house = 2
    d = assess.house_price_vs_number_of_features_coordinates(conn, longitude, latitude, property_type, box_width,
                                                             box_height,
                                                             distance_from_house, year, features)
    d = d.astype({"longitude": float, "lattitude": float})
    d['constant'] = 1
    train, test = train_test_split(d, test_size=0.1)
    column_names = []
    for feature, tags in features.items():
        if len(tags) == 0:
            column_names.append(f"{feature}_number")
        else:
            for tag in tags:
                column_names.append(f"{feature}_{tag}_number")
    column_names.append("constant")
    design = train[column_names]
    y = train['price']
    m_linear_basis = sm.GLM(y, design, family=sm.families.Gaussian(link=sm.families.links.log))
    results_basis = m_linear_basis.fit()
    test_features = test[column_names]
    results = results_basis.get_prediction(test_features).summary_frame(alpha=0.05)['mean']
    test['prediction'] = results
    prediction_features = assess.get_test_features_without_distance(longitude, latitude, box_width, box_height,
                                                                    distance_from_house, year, features)
    prediction_features['constant'] = 1
    prediction_features = prediction_features[column_names]
    predicted_price = results_basis.get_prediction(prediction_features).summary_frame(alpha=0.05)['mean']
    print(results_basis.summary())
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.scatter(test['price'], test['prediction'])
    plt.plot()
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    ax = plt.gca()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    print(test[['price', 'prediction']].head().to_string())
    rmse = mean_squared_error(test['price'], test['prediction'], squared=False)
    mape = mean_absolute_percentage_error(test['price'], test['prediction'])
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Percentage Error: ", mape * 100)
    print("Predicted Price: ", predicted_price)
    return predicted_price


def predict_price_with_distance(conn, latitude, longitude, year, property_type):
    """ Trains a model which then produces a price prediction for a house taking the parameters given.
    Model only takes into account the number of each feature surrounding the house as well as the distance to each
    feature for each house (e.g. distance to closest school).
    See notebook for more specific details about the model.

    Args:
       conn: connection to MariaDB database
       latitude: latitude of house
       longitude: longitude of house
       year: year for which house price should be predicted
       property_type: type of property e.g. 'D' for detached house
    """
    features = {"amenity": ["cafe", "restaurant", "school", "college", "bar"],
                "public_transport": [],
                "leisure": ["park"]}
    distance_features = {"amenity": ["cafe", "restaurant"],
                         "public_transport": [],
                         "leisure": ["park"]}
    box_width = 4
    box_height = 4
    distance_from_house = 2
    column_names = []
    distance_column_names = []
    for feature, tags in features.items():
        if len(tags) == 0:
            column_names.append(f"{feature}_number")
        else:
            for tag in tags:
                column_names.append(f"{feature}_{tag}_number")
    for feature, tags in distance_features.items():
        if len(tags) == 0:
            column_names.append(f"distance_from_{feature}")
            distance_column_names.append(f"distance_from_{feature}")
        else:
            for tag in tags:
                column_names.append(f"distance_from_{feature}_{tag}")
                distance_column_names.append(f"distance_from_{feature}_{tag}")
    column_names.append("constant")
    d = assess.house_price_vs_number_of_features_coordinates(conn, longitude, latitude, property_type,
                                                             box_width, box_height, distance_from_house,
                                                             year, features)
    d = d.astype({"longitude": float, "lattitude": float})
    s = assess.house_price_vs_distance_from_feature_coordinates(conn, longitude, latitude,
                                                                property_type, box_width, box_height,
                                                                year, distance_features)
    a = d.join(s[set(distance_column_names).intersection(set(s.columns))])
    a['constant'] = 1
    train, test = train_test_split(a, test_size=0.1)
    feature_cols = set(column_names).intersection(set(train.columns))
    design = train[feature_cols]
    y = train['price']
    m_linear_basis = sm.GLM(y, design, family=sm.families.Gaussian(link=sm.families.links.log))
    results_basis = m_linear_basis.fit()
    test_features = test[feature_cols]
    results = results_basis.get_prediction(test_features).summary_frame(alpha=0.05)['mean']
    test['prediction'] = results
    prediction_features = assess.get_test_features_with_distance(longitude, latitude, box_width, box_height, distance_from_house,
                                                                 year,
                                                                 features, distance_features)
    prediction_features['constant'] = 1
    prediction_features = prediction_features[feature_cols]
    predicted_price = results_basis.get_prediction(prediction_features).summary_frame(alpha=0.05)['mean']
    print(results_basis.summary())
    prediction_features['pred'] = predicted_price
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.scatter(test['price'], test['prediction'])
    plt.plot()
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    ax = plt.gca()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    print(test[['price', 'prediction']].head().to_string())
    rmse = mean_squared_error(test['price'], test['prediction'], squared=False)
    mape = mean_absolute_percentage_error(test['price'], test['prediction'])
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Percentage Error: ", mape * 100)
    print("Predicted Price: ", predicted_price)
    return predicted_price
