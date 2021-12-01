# This file contains code for suporting addressing questions in the data
import housing_fynesse.assess as assess
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""


def predict_price_without_distance(conn, latitude, longitude, year, property_type):
    features = {"amenity": ["cafe", "restaurant", "school", "college", "bar"],
                "public_transport": [],
                "shop": [],
                "leisure": ["park"]
                }
    box_width = 4
    box_height = 4
    distance_from_house = 1
    d = assess.house_price_vs_number_of_features_coordinates(conn, longitude, latitude, property_type, box_width,
                                                             box_height,
                                                             distance_from_house, year, features)
    d = d.astype({"longitude": float, "lattitude": float})
    if len(d) == 0:
        box_width = 40
        box_height = 40
        distance_from_house = 3
        d = assess.house_price_vs_number_of_features_coordinates(conn, longitude, latitude, property_type, box_width,
                                                                 box_height, distance_from_house, year, features)
        d = d.astype({"longitude": float, "lattitude": float})
    d['constant'] = 1
    train, test = train_test_split(d, test_size=0.2)
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
    prediction_features = assess.number_of_features_surrounding_test(longitude, latitude, box_width, box_height,
                                                                     distance_from_house, year, features)
    prediction_features['constant'] = 1
    prediction_features = prediction_features[column_names]
    predicted_price = results_basis.get_prediction(prediction_features).summary_frame(alpha=0.05)['mean']
    print("Predicted Price is: ", predicted_price)
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
    rmse = mean_squared_error(test['price'], test['prediction'], squared=False)
    mape = mean_absolute_percentage_error(test['price'], test['prediction'])
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Percentage Error: ", mape)
    return test


def predict_price_with_distance(conn, latitude, longitude, year, property_type):
    features = {"amenity": ["cafe", "restaurant", "school", "college", "bar"],
                "public_transport": []}
    distance_features = {"amenity": ["cafe", "restaurant"],
                         "public_transport": []}
    column_names = []
    distance_column_names = []
    box_width = 4
    box_height = 4
    distance_from_house = 1
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
    train, test = train_test_split(a, test_size=0.2)
    feature_cols = set(column_names).intersection(set(train.columns))
    design = train[feature_cols]
    y = train['price']
    m_linear_basis = sm.GLM(y, design, family=sm.families.Gaussian(link=sm.families.links.log))
    results_basis = m_linear_basis.fit()
    test_features = test[feature_cols]
    results = results_basis.get_prediction(test_features).summary_frame(alpha=0.05)['mean']
    test['prediction'] = results

    prediction_features = assess.get_test_features(longitude, latitude, box_width, box_height, distance_from_house,
                                                   year,
                                                   features, distance_features)
    prediction_features['constant'] = 1
    prediction_features = prediction_features[feature_cols]
    predicted_price = results_basis.get_prediction(prediction_features).summary_frame(alpha=0.05)['mean']
    print("Predicted Price: ", predicted_price)
    prediction_features['pred'] = predicted_price
    print(results_basis.summary())
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.xlim([0, 1000000])
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
    rmse = mean_squared_error(test['price'], test['prediction'], squared=False)
    mape = mean_absolute_percentage_error(test['price'], test['prediction'])
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Percentage Error: ", mape)
    return test
