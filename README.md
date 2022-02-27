# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project utilizes Machine Learning to identify credit card customers that are most likely to churn. The project includes all the EDA and feature engineering required to create and train the models. Two models are exported under '/models' directory that use Random Forest Classifier and Logistic Regression. In addition, model evaluation results and scores are available under '/images/result'.  
This project all contains complete testing and logging that can be found in `churn_script_logging_and_tests` file.   

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