- [1. Introduction](#1-introduction)
- [2. Exploration and Visualizations](#2-exploration-and-visualizations)
    - [1. Fetch Data](#1-fetch-data)
# 1. Introduction
The goal of this project is to build a Natural Language Processing multi-class Classifier using Tensorflow, Keras, Hugging Face,   
and other libraries that the City of San Francisco could use to classify complaints received by the building department so the   
complaints could be forwarded to the proper division, such as the Electrical Services Divison or the Building Inspection Division.  

I found the dataset on https://datasf.org/ which is a collection of public datasets the City of San Francisco has made available.   
The dataset consists of 184,593 different complaints that the Building Department has received and include other information   
about the complaint, including the division it was assigned to which will be the target for our model.

The dataset is continously updated as the department receives more complaints and can easily be downloaded using the city's Socrata API.   

Currently the project is utilizing following libraries:  
<ul>
    <li>Tensorflow</li>
    <li>Keras</li>
    <li>Hugging Face</li>
    <li>NLTK</li>
    <li>SpaCy</li>
    <li>Scikit-Learn</li>
<ul>

# 2. Exploration and Visualizations
We will begin by exploring the dataset we will be working with to get an idea of the number of complaints for each department,  
how many null values and duplicates there are, what words are most frequent throughout the dataset and different deaparments,  
and other information about the corpus, or body of text we will be working with. 

### 1. Fetch Data
To begin we fetch our data using the fetch_data module in src, this simply fetches the data from the City of SF's API using our  
API key and stores it in a Pandas dataframe. You must sign up for a key before you can fetch information from the API.

fetch_data definition:
``` python 
def fetch_data(app_key):
    client = Socrata("data.sfgov.org", app_key)
    results = client.get("gm2e-bten", limit = 300000)
    results_df = pd.DataFrame.from_records(results, index ='complaint_number')
    results_df.to_csv(data_path + 'raw_data.csv')
```
Fetching Data:
``` python
from src.config import app_key
fetch_data(app_key)
```
