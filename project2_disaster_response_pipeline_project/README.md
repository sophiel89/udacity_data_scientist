# Disaster Response Pipeline Project

![Picture](screenshots/app.png)


### Table of Contents

1. [Project Summary](#summary)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. 
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Summary<a name="summary"></a>

This is the second project of Udacity Nanodegree "Data Scientist".
The data for this project was was provided by [Figure Eight](https://www.figure-eight.com/) and contains emergency messages from the past and categories for the respective messages (e.g. fire, flood, child alone, etc.).
I used this data (messages and repective categories) to train a machine learning pipeline for the categorization of new emergency messages coming in. 
The resulting model was then visualized using a web app in order to make it more accessible. The web app contains a user interface where the user (e.g. emergency staff) can enter a new emergency message and receives a categorization based on the machine learning model immediately. 
This project is of very high relevance, not only in theory but in real life as it can literally save lifes. In case of an emergency every second counts and processes need to be automated to a maximum. Reducing the time between an incoming emergency message and help arriving is crucial here. This model can help to accomplish that. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Installation <a name="installation"></a>

I used Anaconda Navigator 2.5.1 (including python 3.11.5) throughout this project. 
The following libraries are used, most of them are already included in Anaconda.  

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


## File Descriptions <a name="files"></a>

The following files were used/created throughout the project: 

    - app
        | - template
            | |- master.html  # starting page of web app
            | |- go.html  # result page of web app
        |- run.py  # Flask file running the app

    - data
        |- disaster_categories.csv  # data to process 
        |- disaster_messages.csv  # data to process
        |- process_data.py # ETL pipeline to process data

    - models
        |- train_classifier.py # machine learning pipeline training the model

    - screenshots
        |- app.png

    - README.md

~~~~~~~
        project2_disaster_response_pipeline_project
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_categories.csv
                |-- disaster_message.csv
                |-- process_data.py
          |-- models
                |-- train_classifier.py
          |-- screenshots
                |-- app.png
          |-- README.md
~~~~~~~
## Project Motivation<a name="motivation"></a>

Second project of Udacity Nanodegree "Data Scientist".
Topic: Use data provided by "Figure Eight" containing emergency messages and categories for the relevant messages to train a machine learning pipeline. The trained model should then be applied to categorize new emergency messages, to facilitate and speed up the process to find adequate help in an emergency. The last step is building a web user interface where an emergency message can be entered which es then categorized based on the machine learning model built before. 


## Results<a name="results"></a>

After running the three python files 1) process_data.py 2) train_classifier.py and 3) run.py including the relevant information on input and output paths the user should receive the url to open the web app containing some visualizations on the starting page. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you Figure Eight for making the data available to public. Also thanks to Udacity GPT for helping me with various syntax errors.
