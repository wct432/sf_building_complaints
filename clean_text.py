import string
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def clean(text, exclude_nums=True):

	#lowercase 
	text = text.lower()

	#remove punctuation and numbers if exlude_nums = True
	if exclude_nums:
	   exclude_list = string.punctuation + string.digits
	else:
	   exclude_list = string.punctuation
	translator = str.maketrans('','', exclude_list)
	text = text.translate(translator)

	#tokenize
	pass

	return text

print('success')