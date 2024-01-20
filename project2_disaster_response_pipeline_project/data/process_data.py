"""
Udacity Nanodegree 'Data Scientist'
Project 2: Disaster Response Pipeline

Input: 'disaster_messages.csv' and 'disaster_messages.csv'
Output: 'data_processed_etl.db'

1) In this file both import files are imorted and the data is cleane and processed
2) Afterwards I check for duplicates.
3) Finally, I save the data in an sqlite db to be used in a leter step
"""

# import all necessary libraries for the following steps
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages and cagtegories datasets from csv filepaths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # split categories into separate category columns
    categories = categories['categories'].str.split(';', expand = True)

    # change name of categories-columns based on first row of df
    ## get first row in a list
    column_names = list(categories.iloc[0])
    ## remove two last strings
    category_colnames = list(map(lambda col: col[:-2],  column_names))
    ## update column names in df categories
    categories.columns = category_colnames


    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1].astype(int)

    df = pd.concat([messages, categories], axis = 1)
    return df

def clean_data(df):
#drop duplicate rows from database_filepath
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
# export data and save it in sqlite db
    engine = create_engine('sqlite:///data_processed_etl.db')
    df.to_sql('data_processed_etl', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
