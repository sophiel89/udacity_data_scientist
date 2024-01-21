"""
Udacity Nanodegree 'Data Scientist'
Project 2: Disaster Response Pipeline

Input: preprocessed data 'data/data/data_processed_etl.db'
Output: final classifier modelmodel_final.pkl

1) First, I import the sqlite db created in the last step
2) Next, I tokenize the data as it contains text snippept with error messages
3) Afterwards, I build a machine learning model, test it and and export the model
"""

import sys
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import re
import pickle

# load data from sqlite database
def load_data(database_filepath):

    """
    Arguments:
        database_filepath = path to preprocessed data from last step
    Returns:
        clean_tokens = tokenized text string
    """

    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    df = pd.read_sql("SELECT * FROM data_processed_etl", engine)
    X = df.message.values
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


#tokenize input data using regular expressions to replace urls and WorldNetTokenizer
def tokenize(text, X):

    """
    Arguments:
        text = Input text to tokenize
    Returns:
        X = predictor dataset
        Y = target variables dataset
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls=[]
    for x in X:
        if re.findall(url_regex, x) != []:
            detected_urls.append(re.findall(url_regex, x))

    # I couln't manage to exluce closing brackets at the end of the string with
    # a regular experssion >> use replace method here to account for those

    for url in detected_urls:
        for i in range(len(url)):
            if url[i][-2:] == "))":
                url[i] = url[i].replace("))","")
            elif url[i][-2:] == ").":
                url[i] = url[i].replace(").","")
            elif url[i][-1] == ")":
                url[i] = url[i].replace(")","")
            elif url[i][-1] == "]":
                url[i] = url[i].replace("]","")
            elif url[i][-1] == ".":
                url[i] = url[i].replace(".","")
            elif url[i][-1] == ":":
                url[i] = url[i].replace(":","")

    for url in detected_urls:
        for i in range(len(url)):
            text = text.replace(url[i], "urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build machine learning pipeline using Random Forest as classifier and Multi Output to account for 36 output columns
    text_transformer = Pipeline([
                                ('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer())
                                ])

    feature_union = FeatureUnion([
                                ('text_transformer', text_transformer)
                                ])

    classifier = RandomForestClassifier()
    multi_classifier = MultiOutputClassifier(classifier)

    pipeline = Pipeline([
                        ('features', feature_union),
                        ('clf', multi_classifier)
                        ])

    parameters = {
                    'clf__estimator__n_estimators': [40, 50],
                    'clf__estimator__min_samples_leaf':[2, 5]
                    }

    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Arguments:
    model = model trained with machine learning
    X_test = data set containing predictor
    Y_test = data set containig target variable
    category_names = column names of variables

    Returns: Prints Confusion Matrx, Accuracy Score & F1 Score for each column
    """
    Y_pred = model.predict(X_test)

    for i, column_name in enumerate(Y_test.columns):

        Y_test_col = Y_test.iloc[:,i]
        Y_pred_col = Y_pred[:,i]
        labels = np.unique(Y_pred_col)
        confusion_mat = confusion_matrix(Y_test_col, Y_pred_col, labels=labels)
        accuracy = (Y_pred_col == Y_test_col).mean()
        if np.any(Y_pred_col):
            f1 = f1_score(Y_test_col, Y_pred_col, average='weighted')
        else:
            f1 = 0.0
        report = classification_report(Y_test_col, Y_pred_col)


        print("Column Name:", column_name)
        print("Labels:", labels)
        #print("Confusion Matrix:\n", confusion_mat)
        #print("Accuracy:", accuracy)
        #print("F1 Score:", f1)
        print("Classification Report", report)
        print("\n")


def save_model(model, model_filepath):
    """
    Arguments:
    model = name of final model
    model_filepath = path to save final model
    """
    # Save the model as a pickle file
    with open(model_filepath, 'wb') as path:
        pickle.dump(model, path)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
