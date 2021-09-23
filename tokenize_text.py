from clean_text import clean
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize(text, num_words = 10000,oov_token='<UNK>', pad_type='post', trunc_type='post', maxlen = 200):

    tokenizer = Tokenizer(num_words,oov_token, pad_type, trunc_type)

 	#fit tokenizer to text
    tokenizer.fit_on_texts(text)

 	#define variables for word count and index
    word_count = tokenizer.word_counts
    word_index = tokenizer.word_index

 	#encode the data into a sequence
    train_sequences = tokenizer.texts_to_sequences(text)

 	#pad the sequence
    train_padded = pad_sequences(train_sequences, padding=pad_type,
								truncating=trunc_type, maxlen=maxlen)

    print("Word Count:", word_count)
    print("\n")
    print("Word Index:", word_index)
    print("\n")