# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project utilizes Machine Learning to identify credit card customers that are most likely to churn. The project includes all the EDA and feature engineering required to create and train the models. Two models are exported under '/models' directory that use Random Forest Classifier and Logistic Regression. In addition, model evaluation results and scores are available under '/images/result'.  
This project all contains complete testing and logging that can be found in 'churn_script_logging_and_tests' file.   

## Running Files
Below are the complete steps for running the files in this project:  
1. Before starting, ensure to install all the libraries required for this project by running the command below
    ```
    pip install -r requirements.txt
    ```
    You should see the libraries installing on your computer (make sure you're in the same directory as the requirements file)
    
2. Once you have all the libraries installed,

### Running tests:
This project contain unit tests that confirm the correct behavior for all the functions. Below are the steps to run the test:
1. Run the command below in the root directory of the project (ensure you have `pytest` insalled):
    ```
    pytest
    ```
