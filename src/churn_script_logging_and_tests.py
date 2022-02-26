'''
#######################################################################################
This file contains the tests for churn_library file and logging that outputs test
results to logs/churn_library.log

Author: Ashraf Al Gumaei

Created On: 01/28/2022
#######################################################################################
'''

import os
import logging
from os.path import exists
import pytest
import pandas as pd
import constants
from churn_library import *


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def df():
    '''
    global variable used for testing
    '''
    data = pd.read_csv("./data/bank_data.csv")
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


def test_import():
    '''
    tests import_data function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    tests perform_eda function
    '''
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as e:
        logging.error("Testing perform_eda: Error running the function")
        raise e

    empty_directory = not os.listdir('./images/eda')
    try:
        assert bool(empty_directory) == False
    except AssertionError as err:
        logging.error(
            "Testing perform_eda:	visualization results do not exist in the folder")
        raise err


def test_encoder_helper(df):
    '''
    test encoder helper function
    '''
    try:
        encoded_df = encoder_helper(df, constants.CATEGORY_LIST)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as e:
        logging.error("Testing encoder_helper: Error running the function")
        raise e

    try:
        assert encoded_df.shape[1] == 28
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The output dataframe does not contain right number of columns")
        raise err


def test_perform_feature_engineering(df):
    '''
    test perform_feature_engineering function
    '''
    encoded_df = encoder_helper(df, constants.CATEGORY_LIST)
    try:
        x, x_train, x_test, y_train, y_test = perform_feature_engineering(
            encoded_df, constants.KEEP_COLS)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing perform_feature_engineering: Error running the function")
        raise e

    try:
        assert x.shape[1] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: All/some variables are empty")
        raise err


def test_train_models(df):
    '''
    test train_models function
    '''
    encoded_df = encoder_helper(df, constants.CATEGORY_LIST)
    x, x_train, x_test, y_train, y_test = perform_feature_engineering(
        encoded_df, constants.KEEP_COLS)
    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as e:
        logging.error("Testing train_models: Error running the function")
        raise e

    rfc_model_exists = exists('./models/rfc_model.pkl')
    lr_model_exists = exists('./models/lr_model.pkl')
    try:
        assert bool(rfc_model_exists)
        assert bool(lr_model_exists)
    except AssertionError as err:
        logging.error(
            "Testing train_models: The pickle files do not exist in the directory")
        raise err


def test_classification_report_image(df):
    '''
    test classification_report_image function
    '''
    encoded_df = encoder_helper(df, constants.CATEGORY_LIST)
    _, x_train, x_test, y_train, y_test = perform_feature_engineering(
        encoded_df, constants.KEEP_COLS)
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
        x_train, x_test, y_train, y_test)
    try:
        classification_report_image(
            y_train,
            y_test,
            y_train_preds_rf,
            y_test_preds_rf,
            y_train_preds_lr,
            y_test_preds_lr)
        logging.info("Testing classification_report_image: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing classification_report_image: Error running the function")
        raise e

    rf_report_exists = exists('./images/results/Random_Forest_Report.png')
    lr_report_exists = exists(
        './images/results/Logistic_Regression_Report.png')
    try:
        assert bool(rf_report_exists)
        assert bool(lr_report_exists)
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: Report images do not exist in the directory")
        raise err


def test_feature_importance_plot(df):
    '''
    test feature_importance_plot function
    '''
    plot_output_pth = "./images/results/Feature_Importances.png"
    encoded_df = encoder_helper(df, constants.CATEGORY_LIST)
    x = perform_feature_engineering(encoded_df, constants.KEEP_COLS)[0]
    rfc_model = joblib.load('./models/rfc_model.pkl')
    try:
        feature_importance_plot(rfc_model, x, plot_output_pth)
        logging.info("Testing feature_importance_plot: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing feature_importance_plot: Error running the function")
        raise e

    feature_importance_plot_exists = exists(
        './images/results/Feature_Importances.png')
    try:
        assert bool(feature_importance_plot_exists)
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: Image does not exist in directory")
        raise err


if __name__ == "__main__":
    pass
