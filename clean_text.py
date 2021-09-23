import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))

def clean(text, exclude_nums=True, lemmatize = False):

	#lowercase text
	text = text.lower()

	#remove punctuation and remove numbers by default
	#pass exlude_nums = False to retain numbers
	words = remove_non_letters(text)

	#tokenize into list of words
	word_list = nltk.word_tokenize(words)

	#remove stop words
	no_stopwords = remove_stopwords(word_list)

	#stem text
	stemmed_words = stem_words(no_stopwords)

	#lemmatize text
	if lemmatize:
		stemmed_words = False
		lemmatized_words = lemmatize_words(no_stopwords)
	
	return stemmed_words or lemmatized_words




#removes punctuation and removes numbers by default
#pass exlude_nums=False to avoid removing numbers 
def remove_non_letters(text, exclude_nums = True):
	if exclude_nums:
		exclude_list = string.punctuation + string.digits	
	else: 
		exclude_list = string.punctuation

	translator = str.maketrans('','', exclude_list)
	words = text.translate(translator)
	return words




#remove stopwords from the text
def remove_stopwords(text):
	no_stopwords = []
	stop_words = stopwords.words('english')
	for word in text:
		if word not in stop_words:
		 no_stopwords.append(word)
	return no_stopwords




#stem words
def stem_words(text):
	stemmer = PorterStemmer()
	stemmed_words = []
	for word in text:
		word = stemmer.stem(word)
		stemmed_words.append(word)
	return stemmed_words




#lemmatize words
def lemmatize_words(text):
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # just keep tagger for lemmatization
	lemmatized_words = []
	for token in text:
		test = nlp(token)
		token = test[0].lemma_
		lemmatized_words.append(token)
	return lemmatized_words
