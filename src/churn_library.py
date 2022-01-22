# library doc string


# import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging 
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'

# Set logging 
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    try:
        df = pd.read_csv(pth)
        logging.info("SUCCESS: Your file has been loaded")
        return df
    except FileNotFoundError:
        logging.warning("ERROR: File was not found")


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df.shape
    df.isnull().sum()
    df.describe()
    
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig('./images/eda/Churn.png')

    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/Customer_Age.png')

    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/Marital_Status.png')
    
    plt.figure(figsize=(20,10)) 
    sns.distplot(df['Total_Trans_Ct']);
    plt.savefig('./images/eda/Total_Trans_Ct.png')
    
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/Data_Heatmap.png')
    

def encoder_helper(df, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    
    for category in category_lst:
        category_lst = []
        category_groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            category_lst.append(category_groups.loc[val])

        df['{}_{}'.format(category,response)] = category_lst    
    new_columns_df = df
    return new_columns_df
        
def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    # set x and y
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    y = df['Churn']
    
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Save scores and report to a txt file
#     with open("./data/results.txt", "a") as f:
#         print('Random Forest Results:')
#         print('Test Results')
#         print(classification_report(y_test, y_test_preds_rf))
#         print('Train Results')
#         print(classification_report(y_train, y_train_preds_rf))

#         print('Logistic Regression Results:')
#         print('Test Results')
#         print(classification_report(y_test, y_test_preds_lr))
#         print('Train Results')
#         print(classification_report(y_train, y_train_preds_lr))
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    
    # Save ROC plots to images folder 
#     lrc_plot = plot_roc_curve(lrc, X_test, y_test)
#     plt.figure(figsize=(15, 8))
#     ax = plt.gca()
#     rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
#     lrc_plot.plot(ax=ax, alpha=0.8)
#     plt.savefig('./images/eda/Customer_Age.png')

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    eda = perform_eda(df)
    new_df = encoder_helper(df,['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'])
    y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = perform_feature_engineering(new_df)
    classification_report_image(y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr)