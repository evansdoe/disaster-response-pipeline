import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
import re

# download nltk libraries
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, \
f1_score, precision_score, recall_score, make_scorer

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
	"""
	INPUT:
		- database_filepath - Python str object - path to the database file DisasterResponse.db
		
	OUTPUT:
		- X - Numpy array object - A vector of str objects
		- Y - Numpy array object - A matrix of zeros and ones.
	"""
	
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Load the data from the database
    df = pd.read_sql_table('disaster_categories', engine)
    
    # Get the values of the column labelled messages in the DataFrame
    X = df.message.values
    
    # Get the values in the other columns apart from the following columns: id, messages, 
    # original and genre.
    
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
	
	category_names = np.array(df.drop(columns=['id', 'message', 'original', 'genre']).columns)
	
	return X, Y, category_names
	

def replace_urls(text):
    """
    Returns an editted version of the input Python str object `text` with all urls in 
    text replaced with the str 'urlplaceholder'.
    
    INPUT:
        - text - Python str object - A raw text data
        
    OUTPUT:
        - text - Python str object - An editted version of the input data `text` with all 
        urls in text replaced with the str 'urlplaceholder'.
    """
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    return text	

def tokenize(text):
    """
    Takes a Python string object and returns a list of processed words 
    of the text.
    
    INPUT:
        - text - Python str object - A raw text data
        
    OUTPUT:
        - stem_words - Python list object - A list of processed words from the input `text`.
    """
        
    text = replace_urls(text)
        
    # Text normalising process: 
    # 1. Remove punctuations and 
    # 2. Covert to lower case 
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()

    # tokenize text: 
    # That is, split the text into a list of words
    tokens = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    
    # Reduce words in lemmed to their stems
    stem_words = [PorterStemmer().stem(w) for w in lemmed]

    return stem_words


def build_model():
	"""
	INPUT:
		- None
		
	OUTPUT:
		- pipeline - A machine learning pipeline
	"""
	
    # model pipeline
	pipeline = Pipeline([
    ('features', FeatureUnion([
        
        ('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                                    ('tfdif', TfidfTransformer())
                                    ])),
        
        ('disaster_words', DisasterWordExtrator())
        ])), 
       
    ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(n_jobs=-1)))
    ])
    
    parameters = {
    	'clf__estimator__max_depth': [25, 50], 
    	'clf__estimator__n_estimators': [100, 250]
	}
              	 
	# create grid search object
	clf_rfc_feat_union = GridSearchCV(
    	rfc_feat_union_pipeline, 
    	param_grid=parameters, 
    	scoring=average_accuracy_score, 
    	verbose=10, 
    	return_train_score=True
    )
    
    return pipeline
    
def get_scores(y_true, y_pred):
    """
    Returns the accuracy, precision and recall and f1 scores of the two same shape numpy 
    arrays `y_true` and `y_pred`.

    INPUTS:
        y_true - Numpy array object - A (1 x n) vector of true values
        y_pred - Numpy array object - A (1 x n) vector of predicted values
        
    OUPUT:
        dict_scores - Python dict - A dictionary of accuracy, precision and recall and f1 
        scores of `y_true` and `y_pred`.
    """
    
    # Compute the accuracy score of y_true and y_pred
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute the precision score of y_true and y_pred
    precision = round(precision_score(y_true, y_pred, average='micro'))
    
    # Compute the recall score of y_true and y_pred
    recall = recall_score(y_true, y_pred, average='micro')
    
    # Compute the recall score of y_true and y_pred
    f_1 = f1_score(y_true, y_pred, average='micro')
    
    # A dictionary of accuracy, precision and recall and f1 scores of `y_true` and `y_pred`
    dict_scores = {
        'Accuracy': accuracy, 
        'Precision': precision, 
        'Recall': recall, 
        'F1 Score': f_1
    }
    
    return dict_scores


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    	- model - A machine learning pipeline
    	- X_test - Numpy array - A vector of str objects
    	- Y_test - Numpy array - A matrix of zeros and ones.
    	- category_names - Numpy array - A vector of str objects
    	
    OUTPUT:
    	- df - Pandas DataFrame - A DataFrame of the accuracy, precision, recall, and 
    	f1 scores for each category in category_names. 
    """
    
    # Get the best model estimator 
    model = model.best_estimator_ 
    
    # Use the model to get the predicted Y-values
    
    Y_pred = model.predict(X_test)
    
    # Form a DataFrame of the accuracy, precision, recall, and f1 scores for each 
    #category in category_names.
    
    df = pd.DataFrame([get_scores(Y_test[:, idx], Y_pred[:, idx]) for idx in \
                   range(Y_test.shape[-1])], index=category_names)
    
    print(df.mean())
    print() 
    
    return df
    
    


def save_model(model, model_filepath):
	
    # save the model to disk
    model = model.best_estimator_
    pickle.dump(model, open(model_filepath, 'wb'))


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
