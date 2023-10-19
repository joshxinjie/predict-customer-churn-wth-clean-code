"""
This module contains the unit tests for the churn model functions

Author: Xin Jie Lee
Date: 19 Oct 2023
"""

import os
import logging
import joblib

import pandas as pd
from pytest_cases import parametrize_with_cases

import churn_library as cls

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True
)


def sampled_db():
    """Creates the test case which is a sample of the original raw data

    Returns:
        pd.DataFrame: The sampled DataFrame of the raw data for testing
    """
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 21, 22, 23, 24, 25, 26, 27, 28],
            "CLIENTNUM": [
                768805383, 818770008, 708508758, 784725333, 811604133,
                789124683, 771071958, 720466383, 804424383, 718813833
            ],
            "Attrition_Flag": [
                "Existing Customer", "Existing Customer", "Attrited Customer",
                "Existing Customer", "Existing Customer", "Existing Customer",
                "Existing Customer", "Existing Customer", "Existing Customer",
                "Existing Customer"
            ],
            "Customer_Age": [45, 49, 62, 41, 47, 54, 41, 59, 63, 44],
            "Gender": ["M", "F", "F", "M", "F", "M", "F", "M", "M", "F"],
            "Dependent_count": [3, 5, 0, 3, 4, 2, 3, 1, 1, 3],
            "Education_Level": [
                "High School", "Graduate", "Graduate", "High School",
                "Unknown", "Unknown", "Graduate", "High School",
                "Unknown", "Uneducated"
            ],
            "Marital_Status": [
                "Married", "Single", "Married", "Married", "Single",
                "Married", "Single", "Unknown", "Married", "Single"
            ],
            "Income_Category": [
                "$60K - $80K", "Less than $40K", "Less than $40K",
                "$40K - $60K", "Less than $40K", "$80K - $100K",
                "Less than $40K", "$40K - $60K", "$60K - $80K",
                "Unknown"
            ],
            "Card_Category": [
                "Blue", "Blue", "Blue", "Blue", "Blue",
                "Blue", "Blue", "Blue", "Blue", "Blue"
            ],
            "Months_on_book": [39, 44, 49, 33, 36, 42, 28, 46, 56, 34],
            "Total_Relationship_Count": [5, 6, 2, 4, 3, 4, 6, 4, 3, 5],
            "Months_Inactive_12_mon": [1, 1, 3, 2, 3, 2, 1, 1, 3, 2],
            "Contacts_Count_12_mon": [3, 2, 3, 1, 2, 3, 2, 2, 2, 0],
            "Credit_Limit": [
                12691.0, 8256.0, 1438.3, 4470.0, 2492.0,
                12217.0, 7768.0, 14784.0, 10215.0, 10100.0
            ],
            "Total_Revolving_Bal": [
                777, 864, 0, 680, 1560,
                0, 1669, 1374, 1010, 0
            ],
            "Avg_Open_To_Buy": [
                11914.0, 7392.0, 1438.3, 3790.0, 932.0,
                12217.0, 6099.0, 13410.0, 9205.0, 10100.0
            ],
            "Total_Amt_Chng_Q4_Q1": [
                1.335, 1.541, 1.047, 1.608, 0.573,
                1.075, 0.797, 0.921, 0.843, 0.525
            ],
            "Total_Trans_Amt": [
                1144, 1291, 692, 931, 1126,
                1110, 1051, 1197, 1904, 1052
            ],
            "Total_Trans_Ct": [42, 33, 16, 18, 23, 21, 22, 23, 40, 18],
            "Total_Ct_Chng_Q4_Q1": [
                1.625, 3.714, 0.6, 1.571, 0.353,
                0.75, 0.833, 1.3, 1.0, 1.571
            ],
            "Avg_Utilization_Ratio": [
                0.061, 0.105, 0.0, 0.152, 0.626,
                0.0, 0.215, 0.093, 0.099, 0.0
            ]
        }
    )
    return df


def test_import_data():
    """Test the import_data function

    Raises:
        err: _description_
        err: _description_
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and \
            columns"
        )
        raise err


@parametrize_with_cases("df", cases=sampled_db)
def test_perform_eda(df):
    """Test perform_eda function

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    try:
        df = cls.generate_churn_label(df)
        cls.perform_eda(df)
        assert os.path.exists("images/eda/churn_hist.png")
        assert os.path.exists("images/eda/cust_age_hist.png")
        assert os.path.exists("images/eda/marital_stat_count.png")
        assert os.path.exists("images/eda/tot_trans_hist.png")
        assert os.path.exists("images/eda/corr.png")
        logging.info("Testing eda: SUCCESS. ALL EDA plots successfully saved")
    except AssertionError:
        logging.error("Testing eda: FAILED. Some EDA plots not saved")


@parametrize_with_cases("df", cases=sampled_db)
def test_encoder_helper(df):
    """Test encoder helper function

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    cols_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
    ]
    encoded_columns = [
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"
    ]

    try:
        df = cls.generate_churn_label(df)
        df = cls.encoder_helper(
            df=df,
            category_lst=cols_to_encode,
            label="Churn"
        )
        logging.info(
            "Testing encoder_helper: Successfully ran cols_to_encode")
        assert set(encoded_columns).issubset(set(df.columns))
        logging.info(
            "Testing encoder_helper: SUCCESS. All encoded columns are present"
        )
    except AssertionError:
        missing_col = set(encoded_columns).difference(set(df.columns))
        logging.info(set(df.columns))
        logging.error(
            f"Testing encoder_helper: FAILED. {missing_col} encoded columns \
            are missing"
        )


@parametrize_with_cases("df", cases=sampled_db)
def test_perform_feature_engineering(df):
    """Test the perform_feature_engineering function

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    try:
        (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
            df=df,
            label='Churn'
        )
        # (X_train, X_test, y_train, y_test) = train_test_data
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "Testing perform_feature_engineering: SUCCESS. Training and test \
            features and labels are generated"
        )
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: FAILURE. One or more of \
            the training and test features and labels dataframes are empty."
        )


@parametrize_with_cases("df", cases=sampled_db)
def test_generate_churn_label(df: pd.DataFrame):
    """Test generation of churn label.

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    df = cls.generate_churn_label(df)
    assert df.iloc[0]["Churn"] == 0
    assert df.iloc[1]["Churn"] == 0
    assert df.iloc[2]["Churn"] == 1


@parametrize_with_cases("df", cases=sampled_db)
def test_generate_mean_encoding_marital_status(df):
    """Test the generate_mean_encoding function for the marital status category

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    df = cls.generate_churn_label(df)
    df = cls.generate_mean_encoding(df, 'Marital_Status', 'Churn')
    assert df.iloc[0]["Marital_Status_Churn"] == 0.2
    assert df.iloc[1]["Marital_Status_Churn"] == 0
    assert df.iloc[2]["Marital_Status_Churn"] == 0.2


@parametrize_with_cases("df", cases=sampled_db)
def test_generate_mean_encoding_income_category(df):
    """Test the generate_mean_encoding function for the income category

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    df = cls.generate_churn_label(df)
    df = cls.generate_mean_encoding(df, 'Income_Category', 'Churn')
    assert df.iloc[0]["Income_Category_Churn"] == 0
    assert df.iloc[1]["Income_Category_Churn"] == 0.25
    assert df.iloc[2]["Income_Category_Churn"] == 0.25


@parametrize_with_cases("df", cases=sampled_db)
def test_get_classification_report(df):
    """Test the get_classification_report function

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    rfc_model = joblib.load('./models/rfc_model.pkl')

    try:
        (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
            df=df,
            label='Churn'
        )
        y_train_preds = rfc_model.predict(X_train)
        y_test_preds = rfc_model.predict(X_test)
        cls.get_classification_report(
            y_train=y_train,
            y_test=y_test,
            y_train_preds=y_train_preds,
            y_test_preds=y_test_preds,
            train_report_pth="results/rf_train_report.csv",
            test_report_pth="results/rf_test_report.csv"
        )
        assert os.path.exists("results/rf_train_report.csv")
        assert os.path.exists("results/rf_test_report.csv")
        logging.info(
            "Testing test_get_classification_report: SUCCESS. Train and test \
            classification reports successfully generated"
        )
    except AssertionError:
        logging.error(
            "Testing test_get_classification_report: FAILURE. Train and test \
            classification reports not generated"
        )


@parametrize_with_cases("df", cases=sampled_db)
def test_get_roc_plot(df):
    """Test the get_roc_plot function

    Args:
        df (_type_): _description_
    """
    rf_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
        df=df,
        label='Churn'
    )

    try:
        cls.get_roc_plot(
            rf_model=rf_model,
            lr_model=lr_model,
            X_test=X_test,
            y_test=y_test,
            roc_pth="images/roc_curve.png"
        )
        logging.info(
            "Testing get_roc_plot: SUCCESS. Successfully generated ROC plot"
        )
    except AssertionError:
        logging.error(
            "Testing get_roc_plot: FAILURE. Unable to generate ROC plot"
        )


@parametrize_with_cases("df", cases=sampled_db)
def test_get_feature_importance_plot(df):
    """Test the get_feature_importance_plot function

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    rfc_model = joblib.load('./models/rfc_model.pkl')

    try:
        (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
            df=df,
            label='Churn'
        )
        cls.get_feature_importance_plot(
            rfc_model, X_train, "images/rf_feat_imp.png")
        assert os.path.exists("images/rf_feat_imp.png")
        logging.info(
            "Testing test_feature_importance_plot: SUCCESS. Successfully \
            generated feature importance plot"
        )
    except AssertionError:
        logging.error(
            "Testing test_feature_importance_plot: FAILURE. Unable to \
            generate feature importance plot"
        )


@parametrize_with_cases("df", cases=sampled_db)
def test_train_models(df):
    """Test the train_models function

    Args:
        df (pd.DataFrame): The subset of raw dataset
    """
    try:
        (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
            df=df,
            label='Churn'
        )
        cls.train_models(X_train, X_test, y_train, y_test)
        assert os.path.exists("models/rfc_model.pkl")
        assert os.path.exists("models/logistic_model.pkl")
        logging.info(
            "Testing test_train_models: Models successfully trained and saved."
        )

        assert os.path.exists("results/rf_train_report.csv")
        assert os.path.exists("results/rf_test_report.csv")
        assert os.path.exists("results/lr_train_report.csv")
        assert os.path.exists("results/lr_test_report.csv")
        logging.info(
            "Testing test_train_models: Classification reports generated.")

        assert os.path.exists("images/results/roc_curve.png")
        logging.info("Testing test_train_models: ROC plot generated.")

        assert os.path.exists("images/results/rf_feat_imp.png")
        logging.info(
            "Testing test_train_models: RandomForestClassifier feature \
            importance plot generated."
        )

        logging.info("Testing test_train_models: SUCCESS.")
    except AssertionError:
        logging.error(
            "Testing test_train_models: FAILURE. One or more of model reports \
            are not generated"
        )


if __name__ == "__main__":
    pass
