# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project utilizes Machine Learning to identify credit card customers that are most likely to churn. The project includes all the EDA and feature engineering required to create and train the models. Two models are exported under '/models' directory that use Random Forest Classifier and Logistic Regression. In addition, model evaluation results and scores are available under '/images/result'.  
This project all contains complete testing and logging that can be found in `churn_script_logging_and_tests` file.   
## Files in the Repo
### Structure:

    data
        bank_data.csv
    images
        eda
            Churn.png
            Customer_Age.png
            Data_Heatmap.png
            Marital_Status.png
            Total_Trans_Ct.png
        results
            Feature_Importances.png
            Logistic_Regression_Report.png
            Random_Forest_Report.png
            ROC_curves.png
    logs
        churn_library.log
    models
        lr_model.pkl
        rfc_model.pkl
    churn_library.py
    churn_notebook.ipynb
    churn_script_logging_and_tests.py
    README.md

### Files Description:
This section include a description of all the files and directories: 
- `bank_data.csv`: Data used for training the model 
- `Churn.png`: EDA plot of the churn distribution 
- `Customer_Age.png`: EDA plot of the customer age distribution
- `Data_Heatmap.png`: EDA heat map of the data used for training 
- `Marital_Status.png`: EDA plot of marital status distribution 
- `Total_Trans_Ct.png`: EDA plot of total transaction distribution plot
- `Feature_Importances.png`: Results for feature importance using Logistics Regression model that's trained from the data 
- `Logistic_Regression_Report.png`: Classification report image of the Logistics Regression model created
- `Random_Forest_Report.png`: Classification report image of the Random Forest Classifier model created
- `ROC_curves`: ROC curve results plot created
- `churn_library.log`: Log file contained all the test results 
- `lr_model.pkl`: Logistics regression model created from training 
- `rf_model.pkl`: Random Forest Classifier model created from training 
- `churn_library.py`: Contains the core functions that are used to run the project.
- `churn_script_logging_and_tests.py`: Contains the unit tests and logging for the project.
- `constants.py`: Contains constants that are used by project 
- `churn_notebook.ipynb`: Initial jupiter notebook used for the initial development of the project 


## Prerequisites
1. Before starting, ensure to install all the libraries required for this project by running the command below (make sure you're in the same directory as the requirements file) 
    ```
    pip install -r requirements.txt
    ```
    You should now see the libraries installing on your computer 

## Running tests:
This project contain unit tests that confirm the correct behavior for all the functions. Tests also include logs that otput to `/logs/churn_library.log` that contains the test resukts. Below are the steps to run the tests:
1. Run the command below in the directory where `churn_script_logging_and_test.py` exists in the project:
    ```
    ipython churn_script_logging_and_tests.py
    ```
2. Once the command finishes running, you should see all the test results in the log file 
3. Alternatively, you can run the command below in the `src` directory to see the test results in the terminal (ensure to install pytest)
    ```
    pytest churn_script_logging_and_tests.py
    ```
    This will use `pytest` module to run the tests and display the test results in the terminal