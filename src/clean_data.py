import string
import nltk
from nltk.corpus.reader.nombank import NombankSplitTreePointer
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


#define spacy processing class which will be used for our lemmatize function
#because it is only being used for lemmatizing we don't need the DepencendyParser or EntityRecognizer
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) 

#define stemmer for our stemmer function
stemmer = PorterStemmer()




def normalize_text(text, exclude_nums=True, lemmatize = False):
    """
    Normalize the text by first converting to lowercase, removing punctuation and/or numbers,
    converting words to tokens, removing stopwords, then finally return the stemmed
    or lemmatized text. Stemming is the default as lemmatizing takes significantly longer.

	Args:
		text: The text to normalize.
		exclude_nums: Whether to exclude numbers from the text.
		lemmatize: Controls if the text is lemmatized or stemmed.
	
	Returns:
		The normalized text as a list of tokens.
	
	Raises:
		AttributeError if the text passed is not a string. 
    """
    #lowercase text
    text = lowercase(text)

    #remove punctuation and remove numbers by default
    text_cleaned = remove_non_letters(text)

    #remove stop words
    tokens = tokenize_remove_stopwords(text_cleaned)

    #lemmatize or stem the words
    if lemmatize:
        lemmatized_words = lemmatize_words(tokens)
    else:
        stemmed_words = stem_words(tokens)

    # print("text_cleaned :", text_cleaned)
    # print("\n")
    # print("Tokens without stopwords :", tokens)
    # print("\n")

    return lemmatized_words if lemmatize else stemmed_words




def lowercase(text):
    """
    Transforms text into lowercase.

    Args:
        Text to transform.

    Returns:
        The lowercased text.

    Raises:
        Exception error if text passed in is not a string.
    """
    if not isinstance(text, str):
        raise AttributeError(f'Passed in type {type(text)}, but type should be a string.')
    lowercased = text.lower()
    return lowercased




def remove_non_letters(text,exclude_nums=True):
    """
    Remove punctuation and numbers from text by default
    If exclude_nums = False numbers will not be removed

    Args:
        text: text to clean
        exclude_nums: Whether to remove numbers from the text or not.
                      True by default.
        
    Returns:
        The text with puncutation and/or numbers removed. 

    Raises: 
        AttributeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise AttributeError(f'Passed in type {type(text)}, but type should be a string.')
    if exclude_nums:
        exclude_list = string.punctuation + string.digits	
    else: 
        exclude_list = string.punctuation
    translator = str.maketrans('','', exclude_list)
    words = text.translate(translator)
    return words




def tokenize_remove_stopwords(text):
    """
    Remove stopwords from the text.

    Args: 
        text: text to remove stopwords from

    Returns:
        Text with stopwords removed
    
    Raises: 
        AttributeError: If text is not a string.
    """
    no_stopwords = []
    stop_words = stopwords.words('english')
    tokens  = nltk.word_tokenize(text)
    for token in tokens:
	    if token not in stop_words:
	        no_stopwords.append(token)
    return no_stopwords




def stem_words(text):
    """
    Stem the words in a string using NLTK's PorterStemmer.

    Args: 
        text: The text to stem.
    
    Returns: 
        A list of stemmed words.

    Raises: 
        AttributeError if the text passed in is not a string.
    """
    if not isinstance(text, list):
        raise AttributeError(f'Passed in type {type(text)}, but type should be a list.')
    stemmed_words = []
    for word in text:
        stemmed = stemmer.stem(word)
        stemmed_words.append(stemmed)
    return stemmed_words




def lemmatize_words(text):
    """
    Lemmatize the words instead of stemming.

    Args:
        text: The text to lemmatize.

    Returns:
        A list of lemmatized words.
    
    Raises:
        AttributeError if the text passed in is not a string.
    """
    if not isinstance(text, list):
        raise AttributeError(f'Passed in type {type(text)}, but type should be a list.')
    lemmatized_words = []
    for doc in nlp.pipe([word for word in text]):
	    for token in doc:
		    lemmatized_words.append(token.lemma_)
    return lemmatized_words