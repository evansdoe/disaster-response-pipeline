from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re

# download nltk libraries
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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
	
class DisasterWordExtrator(BaseEstimator, TransformerMixin):

	def contain_disaster_word(self, text):
		"""
		INPUT:
			- text - Python str object - A raw text data

		OUTPUT:
			- bool - Python bool object - True or False
		"""

		# Words that communicates ones necessary need during a disaster.
		# These can be updated as well. 
		dis_words = ['hunger', 
					 'hungry', 
					 'food', 
					 'water',
					 'drink', 
					 'eat',
					 'thirst', 
					 'medicine', 
					 'medicial', 
					 'cloth', 
					 'shelter', 
					 'help'
					]

		# Lemmatise the words in dis_words
		lemmed_dis_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in dis_words]

		# Get the stem words of each word in lemmed_dis_words
		stem_dis_words = [PorterStemmer().stem(w) for w in lemmed_dis_words]
        
		# Replace all urls in the input str object text
		text = replace_urls(text)

		# Tokenise the str object text
		stem_words = tokenize(text)

		# return whether stem_words contains any of words in stem_dis_words
		return any([words in stem_dis_words for words in stem_words])


	def fit(self, X, y=None):

		return self

	def transform(self, X):

		X_dis_word = pd.Series(X).apply(self.contain_disaster_word)
        
		return pd.DataFrame(X_dis_word)
