"""
This module contains the functions for the churn model

Author: Xin Jie Lee
Date: 19 Oct 2023
"""


# import libraries
import os
from typing import Tuple

import joblib
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    """Returns dataframe for the csv found at pth

    Args:
        pth (str): a path to the csv

    Returns:
        df (pd.DataFrame): pandas dataframe
    """
    df = pd.read_csv(pth)

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Perform eda on df and save figures to images folder

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        None
    """
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig("images/eda/churn_hist.png")

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig("images/eda/cust_age_hist.png")

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("images/eda/marital_stat_count.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("images/eda/tot_trans_hist.png")

    corr_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Churn'
    ]

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df[corr_cols].corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig("images/eda/corr.png")


def generate_mean_encoding(
        df: pd.DataFrame,
        col: str,
        label: str
) -> pd.DataFrame:
    """Generate mean encoding for the given column of the data.

    Args:
        df (pd.DataFrame): pandas dataframe
        col (str): the column name

    Returns:
        df (pd.DataFrame): pandas dataframe with the new mean encodings column
    """
    mean_encodings = df.groupby([col])[label].mean().to_dict()
    df[f'{col}_Churn'] = df[col].map(mean_encodings)

    return df


def encoder_helper(
        df: pd.DataFrame,
        category_lst: list,
        label: str
) -> pd.DataFrame:
    """Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook.

    Args:
        df (pd.DataFrame): pandas dataframe
        category_lst (list): list of columns that contain categorical features
        label (str): string of label name [optional argument that could be used
            for naming variables or index y column]

    Returns:
        df (pd.DataFrame): pandas dataframe with new columns for
    """
    for col in category_lst:
        df = generate_mean_encoding(
            df=df,
            col=col,
            label=label
        )

    return df


def generate_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the churn label based on the attritution flag where Existing
    Customer is treated as non churn while Attrited Customer is considered to
    have churn.

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        df (pd.DataFrame): Pandas dataframe with new churn label column
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_feature_engineering(
    df: pd.DataFrame,
    label: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform feature engineering steps, such as generating churn labels,
    label enocde the categorical features, drop unnecessary columns, run train
    test split.

    Args:
        df (pd.DataFrame): pandas dataframe
        label (str): string of response name [optional argument that could be
        used for naming variables or index y column]

    Returns:
    tuple containing

        - X_train: The train features
        - X_test: The test features
        - y_train: The train features
        - y_test: The test features
    """
    df = generate_churn_label(df)

    cols_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]
    df = encoder_helper(df=df, category_lst=cols_to_encode, label="Churn")

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def get_classification_report(
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        y_train_preds: pd.Series,
        y_test_preds: pd.Series,
        train_report_pth: str,
        test_report_pth: str
) -> None:
    """Produces classification reports for training and testing results and
    stores the reports as csv in the specified paths.

    Args:
        y_train (pd.DataFrame): The train labels
        y_test (pd.DataFrame): The test labels
        y_train_preds (pd.Series): The model train predictions
        y_test_preds (pd.Series): The model test predictions
        train_report_pth (str): The path to store the train classification
            report
        test_report_pth (str): The path to store the test classification report

    Returns:
        None
    """

    test_report = classification_report(
        y_test, y_test_preds, output_dict=True
    )
    test_report = pd.DataFrame(test_report).transpose()
    test_report.to_csv(test_report_pth, index=True)

    train_report = classification_report(
        y_train, y_train_preds, output_dict=True
    )
    train_report = pd.DataFrame(train_report).transpose()
    train_report.to_csv(train_report_pth, index=True)


def get_roc_plot(
    rf_model: RandomForestClassifier,
    lr_model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    roc_pth: str
) -> None:
    """Produces the ROC curve plot and stores the plot in the specified path.

    Args:
        rf_model (RandomForestClassifier): The trained RandomForestClassifier
            model
        lr_model (LogisticRegression): The trained LogisticRegression model
        X_test (pd.DataFrame): The test features
        y_test (pd.Series): The test labels
        roc_pth (str): The path to store the ROC plot

    Returns:
        None
    """
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        rf_model, X_test, y_test, ax=ax, alpha=0.8
    )
    lrc_plot = RocCurveDisplay.from_estimator(lr_model, X_test, y_test)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(roc_pth)


def get_feature_importance_plot(
        model: RandomForestClassifier,
        X_data: pd.DataFrame,
        output_pth: str
) -> None:
    """Creates and stores the feature importances in pth

    Args:
        model (ClassificationModels): The trained sklearn classification model
            objects containing feature_importances_
        X_data (pd.DataFrame): pandas dataframe of X values
        output_pth (str): path to store the figure

    Returns:
        None
    """
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Train, store model results: images + scores, and store models

    Args:
        X_train: The train features
        X_test: The test features
        y_train: The test features
        y_test: The test labels

    Returns:
        None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    get_classification_report(
        y_train=y_train,
        y_test=y_test,
        y_train_preds=y_train_preds_rf,
        y_test_preds=y_test_preds_rf,
        train_report_pth="results/rf_train_report.csv",
        test_report_pth="results/rf_test_report.csv"
    )

    get_classification_report(
        y_train=y_train,
        y_test=y_test,
        y_train_preds=y_train_preds_lr,
        y_test_preds=y_test_preds_lr,
        train_report_pth="results/lr_train_report.csv",
        test_report_pth="results/lr_test_report.csv"
    )

    get_roc_plot(
        rf_model=rfc,
        lr_model=lrc,
        X_test=X_test,
        y_test=y_test,
        roc_pth="images/results/roc_curve.png"
    )

    get_feature_importance_plot(
        model=rfc,
        X_data=X_test,
        output_pth="images/results/rf_feat_imp.png"
    )

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
