# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Anaconda Navigator 2.5.1 (including python 3.11.5)
Libraries used throughout the project: 

    Data Preparation
        - sys
        - pandas
        - sqlalchemy
    Model Estimation
        - nltk
        - sklearn
        - numpy
        - re
        - pickle
    Building Web App
        - json
        - plotly
        - joblib
        - flask

## Project Motivation<a name="motivation"></a>

Second project of Udacity Nanodegree "Data Scientist".
Topic: Use data provided by "Figure Eight" containing emergency messages and categories for the relevant messages to train a machine learning pipeline. The trained model should then be applied to categorize new emergency messages, to facilitate and speed up the process to find adequate help in an emergency. The last step is building a web user interface where an emergency message can be entered which es then categorized based on the machine learning model built before. 

## File Descriptions <a name="files"></a>

The following files were used/created throughout the project: 

    - app
        | - template
            | |- master.html  # main page of web app
            | |- go.html  # classification result page of web app
        |- run.py  # Flask file that runs app

    - data
        |- disaster_categories.csv  # data to process 
        |- disaster_messages.csv  # data to process
        |- process_data.py
        |- InsertDatabaseName.db   # database to save clean data to

    - models
        |- train_classifier.py
        |- classifier.pkl  # saved model 

    - README.md

## Results<a name="results"></a>

After running the three python files 1) process_data.py 2) train_classifier.py and 3) run.py including the relevant information on input and output paths the user should receive the url to open the web app containing some visualizations on the starting page. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you Figure Eight for making the data available to public. Also thanks to Udacity GPT for helping me with various syntax errors.
