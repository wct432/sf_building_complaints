# sf_building_complaints
Natural Language Processing project that classifies complaints received by the San Francisco Department of Building Inspection into the division that needs to address the compliant, such as the Electrical Services Division. The project uses Spacy and the Natural Language Toolkit to preprocess the text using functions defined in clean_data.py. Lemmatized and stemmed versions of the corpus are created and saved in the /data subdirectory. The tokenized text is then fed to a variety of Tensorflow models to compare performance, including Sequential, LSTM, and BERT models. 


Currently this repo is making use of the following frameworks and technologies:
<br>
<ul>
<li>Tensorflow</li>
<li>Hugging Face</li>
<li>NLTK</li>
<li>SpaCy</li>
<li>Scikit-Learn</li>
