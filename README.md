- [1. Introduction](#1-introduction)
- [2. Exploratory Analysis, Pre-Processing, and Visualizations](#2-exploratory-analysis-pre-processing-and-visualizations)
    - [1. Fetch and Load Data](#1-fetch-and-load-data)
    - [2. Explore Data](#2-explore-data)
      - [1. View Head of Data](#1-view-head-of-data)
      - [2. View Shape of Data](#2-view-shape-of-data)
      - [3. Find and Drop Duplicates](#3-find-and-drop-duplicates)
      - [4. Check Data Types](#4-check-data-types)
      - [5. Check Class Imbalance](#5-check-class-imbalance)
      - [6. Select Classes for our Model](#6-select-classes-for-our-model)
      - [7. Dealing with Null Values](#7-dealing-with-null-values)
      - [8. Analyze Complaint Length](#8-analyze-complaint-length)
      - [9. Complaint Length Histogram](#9-complaint-length-histogram)
      - [10. Complaints Wordcloud](#10-complaints-wordcloud)
      - [11. Most Common Collocations](#11-most-common-collocations)
      - [12. Wordcloud for Each Class](#12-wordcloud-for-each-class)
      - [13. Save Pre-Processed Data](#13-save-pre-processed-data)
- [3. Clean and Normalize Data](#3-clean-and-normalize-data)
    - [1. Lemmatizing Text](#1-lemmatizing-text)
    - [2. View Lemmatized Text](#2-view-lemmatized-text)
    - [3. Check for Duplicates After Lemmatizing](#3-check-for-duplicates-after-lemmatizing)
    - [4. View Shape and Class Balance of Lemmatized Data](#4-view-shape-and-class-balance-of-lemmatized-data)
    - [5. Calculate Class Weights](#5-calculate-class-weights)
    - [6. Split Features/Label and Binarize Label](#6-split-featureslabel-and-binarize-label)
    - [7. Create Sequences using Tensorflow Tokenizer](#7-create-sequences-using-tensorflow-tokenizer)
- [5. DistilBERT Model](#5-distilbert-model)
    - [1. Create Train, Test, and Validation Sets](#1-create-train-test-and-validation-sets)
    - [2. Define DistilBERT Tokenizer](#2-define-distilbert-tokenizer)
    - [3. Create DistilBERT Model](#3-create-distilbert-model)
    - [4. Creating Callbacks](#4-creating-callbacks)
        - [1. Step Decay Schedule](#1-step-decay-schedule)
        - [2. Model Checkpoint](#2-model-checkpoint)
        - [3. Early Stopping](#3-early-stopping)
        - [4. TQDM Progress Bar](#4-tqdm-progress-bar)
    - [5. Train Model](#5-train-model)
    - [6. Evaluate Model](#6-evaluate-model)
      - [1. Make Test Predictions](#1-make-test-predictions)
      - [2. Evaluate AUC on Test Data](#2-evaluate-auc-on-test-data)
      - [3. Evaluate Accuracy](#3-evaluate-accuracy)
      - [4. Confusion Matrix](#4-confusion-matrix)
      - [5. Create Classification Report](#5-create-classification-report)
# 1. Introduction
The goal of this project is to build a Natural Language Processing multi-class Classifier using Tensorflow, Keras, Hugging Face,   
and other libraries that the City of San Francisco could use to classify complaints received by the building department so the   
complaints could be forwarded to the proper division, such as the Electrical Services Divison or the Building Inspection Division.  

I found the dataset on https://datasf.org/ which is a collection of public datasets the City of San Francisco has made available.   
The dataset consists of 268,628 different complaints that the Building Department has received and include other information   
about the complaint, including the division it was assigned to which will be the target for our model.

The dataset is continously updated as the department receives more complaints and can easily be downloaded using the city's Socrata  
open Data API, or SODA.  

Currently the project is utilizing following libraries:  
<ul>
    <li>Tensorflow</li>
    <li>Keras</li>
    <li>Hugging Face</li>
    <li>NLTK</li>
    <li>SpaCy</li>
    <li>Scikit-Learn</li>
<ul>

# 2. Exploratory Analysis, Pre-Processing, and Visualizations
We will begin by exploring the dataset we will be working with to get an idea of the number of complaints for each department,  
how many null values and duplicates there are, what words are most frequent throughout the dataset and different deaparments,  
and other information about the corpus, or body of text we will be working with. 


### 1. Fetch and Load Data
To begin we fetch our data using the fetch_data module in src, this simply fetches the data from the City of SF's API using our  
API key and stores it in a Pandas dataframe. You must sign up for a key before you can fetch information from the API.

Definition of fetch_data in src:
``` python 
def fetch_data(app_key):
    client = Socrata("data.sfgov.org", app_key)                                 #define Socrata client 
    results = client.get("gm2e-bten", limit = 300000)                           #fetch our dataset from the SODA API
    results_df = pd.DataFrame.from_records(results, index ='complaint_number')  #store results in Pandas dataframe
    results_df.to_csv(data_path + 'raw_data.csv')                               #save results to /data/ subdirectory
```
Fetching Data:
``` python
from src.config import app_key
fetch_data(app_key)
```

After the data is finished downloading it is stored in our data subdirectory and we can now load it as dataframe using Pandas.   
We will use the columns "complaint_description" and "assigned_division", which will be our model's target.  

Loading data: 
``` python
#load data into a Panda's dataframe
working_dir = os.getcwd()
data_path = os.path.dirname(working_dir) + '/data/'
df = pd.read_csv(data_path + 'raw_data.csv', usecols=['complaint_description', 'assigned_division'])
```

### 2. Explore Data 
Now we are ready to explore our data, we want to get a feel for the dataframe and for the distribution of our data amongst the  
different classes.  

#### 1. View Head of Data
``` python
df.head()
```
<details>
<summary> Click to display dataframe head! </summary>

<p>

|    | complaint_description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | assigned_division            |
|---:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|
|  0 | Caller reporting that the water is discolored. orange color in the sink, water heater(stains) drizzles down the walls;  the exterior windows were painted, the people that were painting wore mask, caller states they were not informed of the particles that were coming in the window. the building was tested for asbestos and the finding were not disclosed to the tenants. caller states the management is doing there own testing, caller states there was a gas leak and they were not given the reason why pgand e was turned off. if elevator is out for a few weeks it makes it hard on the elders in the building to get out. | Housing Inspection Services  |
|  1 | Date last observed: 03-oct-17;    time last observed: 10/6/17--noon;    exact location: rear bldg;    building type: residence/dwelling   illegal unit; work w/o permit; ;    additional information: appears a room is being added to an existing shed.;                                                                                                                                                                                                                                                                                                                                                                                  | Building Inspection Division |
|  2 | Date last observed: 08-dec-17;    exact location: main bldg;    building type: residence/dwelling   illegal unit; ;    additional information: anonymous complaint of illegal units in building;                                                                                                                                                                                                                                                                                                                                                                                                                                           | Housing Inspection Services  |
|  3 | Date last observed: 19-aug-17;    time last observed: shapiro;    exact location: main bldg;    building type: commercial/business   work beyond scope of permit; ;    additional information: again i'm awakened by the sounds of heavy equipment rolling into this site.  4:50am on a saturday.  yesterday was 4:30am.  this can not be legal.;                                                                                                                                                                                                                                                                                          | Building Inspection Division |
|  4 | Date last observed: 14-aug-17;    time last observed: 8 am;    floor: n/a;    unit: n/a;    exact location: main bldg;    building type: residence/dwelling   other building; ;    additional information: contractor is occupying street and storing equipment and dumpsters along entire street front of project and no valid street use permit is displayed - only an expired one;                                                                                                                                                                                                                                                      | Building Inspection Division |

</p>
</details>

#### 2. View Shape of Data
``` python
df.shape
```
*(268628, 2)*

#### 3. Find and Drop Duplicates
View Number of Duplicates:
``` python
#determine how many duplicate entries there are
df.duplicated().sum()
```
*84035*

Drop Duplicates:
``` python
#drop duplicates
df.drop_duplicates(inplace = True)
```

View Shape After Removing Duplicates:
``` python
print("Shape: ", df.shape)
```
*Shape:  (184593, 2)*

#### 4. Check Data Types
Now we will quickly check the datatypes , both columns should be objects.
``` python
#ensure datatypes are correct
df.dtypes
```
*complaint_description    object  
assigned_division        object  
dtype: object*

#### 5. Check Class Imbalance
``` python
#determine how many complaints there are for each category
df.assigned_division.value_counts()
```
*Housing Inspection Services       87118  
Building Inspection Division      63685  
Plumbing Inspection Division      16297  
Code Enforcement Section          10766  
Electrical Inspection Division     5544  
Disabled Access Division           1114  
Help Desk / Technical Services       33  
Department of Bldg Inspection        11  
Other/Outside Agency                  9  
Department of City Planning           4  
Seismic Safety Section                1  
Central Permit Bureau                 1  
Major Plan Check Division             1  
Department of Public Health           1  
Name: assigned_division, dtype: int64*

Clearly we are dealing with an imbalanced data set, with virtually half of the complaints belonging to Housing Inspection Services.   
We will explore different techniques later on to deal with this imbalance, through experimentation I found the best results by using   
class weights in Tensorflow and utilizing transfer learning by using a BERT encoder as the first layer in our final model.  

This gave the model a much deeper understanding of the text, which was especially helpful for classes with less data,   
like the Disabled Access Division.

#### 6. Select Classes for our Model
The Code Enforcment Division doesn't contain complaints the city received, it is  a log of inspections and infractions so we will    
drop it from the database. We are also going to remove any categories that don't have at least 1,000 complaints, as there are many categories with virtually no data. 

Removing Unwanted Classes from Dataframe:
``` python
df = df.groupby("assigned_division").filter(lambda x: len(x) > 1000) #drop any divisions with less than 1000 complaints
df = df.loc[df['assigned_division'] != 'Code Enforcement Section']   #remove Code Enforcement Section
```

Ensure Classes are Correct after Removal:
``` python
#ensure the dataframe is as expected after dropping columns
df.assigned_division.value_counts()
```
*Housing Inspection Services       87118  
Building Inspection Division      63685  
Plumbing Inspection Division      16297  
Electrical Inspection Division     5544  
Disabled Access Division           1114  
Name: assigned_division, dtype: int64*  

#### 7. Dealing with Null Values
Checking for Null Values: 
``` python
#check for null values
df.isna().sum()
```
*complaint_description    4
assigned_division        0
dtype: int64*

With only 4 null values in the complaint_description we will simply drop them from the dataframe:
``` python
#drop rows that contain null values
df.dropna(inplace = True)
```

#### 8. Analyze Complaint Length
Now we will explore the length of our complaints, we want to find the minimum, maximum, and average complaint length,  
as well as view the distribution of the length of complaints. 

Finding Complaint Length Statistics:
``` python
sample = df.copy()                                                          #copy dataframe
sample['complaint_length'] = sample['complaint_description'].apply(len)     #calculate length of each complaint
print("COMPLAINT LENGTH STATISTICS:")
print("Mean Complaint Length: ", sample['complaint_length'].mean())         #print mean complaint length
print("Min Complaint Length: ", sample['complaint_length'].min())           #print minimum complaint length
print("Max Complaint Length: ", sample['complaint_length'].max())           #print maximum complaint length

```
*COMPLAINT LENGTH STATISTICS:  
Mean Complaint Length:  150.45421688133797  
Min Complaint Length:  1  
Max Complaint Length:  1000*   

#### 9. Complaint Length Histogram
Now we will visualize the distribution of the length of our complaints using a histogram. 
<details>
<summary> Click to display code! </summary>

<p>

``` python
#create histogram showing distribution of complaint length
fig, ax = plt.subplots(figsize=(18,8))
ax = sns.histplot(data = sample, x='complaint_length', bins = 100, color = "blue")
xticks = plt.xticks(np.arange(0, 1001, 50.0))
ax.tick_params(axis='x', labelsize=20)
ax.set_xlabel('Complaint Length', fontsize = 30)
ax.set_ylabel('Count', fontsize = 30)
```

</p>
</details>

![Complaint Length Histogram](/../images/images/complaint_length_histogram.png?raw=true)

#### 10. Complaints Wordcloud
Now we will generate a wordcloud to display the most common words in the corpus. 

<details>
<summary> Click to display code! </summary>

<p>

``` python
text = " ".join(complaint for complaint in df.complaint_description)                #join complaints together
wordcloud = WordCloud(stopwords=stop_words, background_color="whitesmoke",          #create wordcloud
                     collocations = False, width=2400, height=1000).generate(text)

fig, ax = plt.subplots(figsize=(24,10))                                             #set figsize
ax = plt.imshow(wordcloud, interpolation='bilinear')                                #dispaly wordcloud
plt.axis("off")                                                                     #remove axis
```

</p>
</details>

![Complaints Wordcloud](/../images/images/complaints_wordcloud.png?raw=true)


#### 11. Most Common Collocations
Now we will display the most common collocations, or pairs of words, found in our corpus. Collocations are powerful   
as they can help us to see patterns throughout the corpus and the pairs can generate more meaning and context than  
a single word alone. 

<details>
<summary> Click to display code! </summary>

<p>

``` python
text = " ".join(complaint for complaint in df.complaint_description)                #join complaints together 
wordcloud = WordCloud(stopwords=stop_words, background_color="aliceblue",           #create collocation wordcloud (pairs of words)
                     colormap = "tab10", width=2400, height=1000).generate(text)

fig, ax = plt.subplots(figsize=(24,10))                                             #set figure size
ax = plt.imshow(wordcloud, interpolation='bilinear')                                #display wordcloud
plt.axis("off")                                                                     #remove axis                                                  
```

</p>
</details>

![Complaints Wordcloud Collocations](/../images/images/complaint_collocations_wordcloud.png?raw=true)

#### 12. Wordcloud for Each Class
Now we will create a wordcloud for each class.

<details>
<summary> Click to display code! </summary>

<p>

``` python
complaint_classes = np.unique(df['assigned_division'].values)                       #create list of class names

fig, axes = plt.subplots(3,2, figsize=(80, 40))                                     #prepare figure with subplots
axes = axes.flatten()                                                               #flatten axes
axes[5].remove()                                                                    #remove 6th element from axes as we only have 5 classes

for i, complaint_class in enumerate(complaint_classes):                             #loop through each class with its index

    text = " ".join(complaint for complaint in df.loc[df['assigned_division'] \
                    == complaint_class].complaint_description)                      #for each class join the complaints together into one text for that class

    wordcloud = WordCloud(stopwords=stop_words,background_color="aliceblue",        #create a wordcloud for the class
                          collocations = False,colormap = "tab10").generate(text)
    
    axes[i].imshow(wordcloud,interpolation='none')                                  #display the class on axes using its index to determine position in subplot
    axes[i].set_title(complaint_class, size = 60)                                   #set title of the wordcloud to the class name
```

</p>

</details>
</summary>

![Wordcloud of Complaints by Division](/../images/images/wordcloud_by_division.png?raw=true)


#### 13. Save Pre-Processed Data
Finally we save our Pre-Processed data, next we will prepare it for our Tensorflow model by lemmatizing the text. 
``` python
#finally we save the preprocessed data to our data subdirectory 
preprocessed = df.copy()
preprocessed.to_csv(data_path + '/preprocessed.csv',index = False)
```




# 3. Clean and Normalize Data
Now we will prepare our text for machine learning so that it can be fed to our Tensorflow model. To prepare the text we must  
perform several operations, we will:
    <ul>
        <li>Remove capitals</li>
        <li>Remove Non-Alphabetic Characters</li>
        <li>Remove Stopwords</li>
        <li>Stem or Lemmatize the text</li>
    </ul>

### 1. Lemmatizing Text
Stemming and lemmatization are techniques used to reduce the inflection of a word and derive a more common base form of the word.

The difference between the two is lemmatization takes the context of words into account when determing the form to  
reduce them to, by taking the context into account the lemmatizer determines the part of speech of the word which helps it to  
accurately reduce the word to its most basic form. However this operation requires a dictionary lookup so lemmatization takes   
significantly longer than stemming. 

Both operations will convert the text into a tokenized representation of each complaint where the complaint is a list of    
the stemmed or lemmatized tokens.   

Refer to the clean_data module in src to review the different helper functions. The normalize_text function groups all of the helper    
functions together and returns a lowercased stemmed or lemmatized text with punctuation, numbers, and stopwords removed.   

``` python
def normalize_text(text, lemmatize = False):
    """
    Normalize the text by first converting to lowercase, removing punctuation and numbers,
    converting words to tokens, removing stopwords, then finally return the stemmed
    or lemmatized text. Stemming is the default behaviour.

	Args:
		text: The text to normalize.
		exclude_nums: Whether to exclude numbers from the text.
		lemmatize: Controls if the text is lemmatized or stemmed.
	
	Returns:
		The normalized text as a list of tokens.
    """
    #lowercase text
    text = lowercase(text)

    #remove punctuation and remove numbers
    text_cleaned = remove_non_letters(text)

    #remove stop words
    tokens = tokenize_remove_stopwords(text_cleaned)

    #lemmatize or stem the words
    if lemmatize:
        lemmatized_words = lemmatize_words(tokens)
    else:
        stemmed_words = stem_words(tokens)

    #return lemmatized or stemmed text
    return lemmatized_words if lemmatize else stemmed_words
```



Now we will normalize our text by applying normalize_text to the dataset with lemmatize = True.

``` python
#use tqdm for progress bar
tqdm.pandas()                                           
#create lemmatized text
lemmatized = df.copy()
#apply normalize_text to dataframe
lemmatized['complaint_description'] = lemmatized['complaint_description'] \
                                      .progress_apply(lambda x: normalize_text(x,
                                                                lemmatize = True))
#save lemmatized version of text
lemmatized.to_csv(lemmatized_path + '/lemmatized_text.csv')
```

I also experimented with using stemmed versions of the text, but I found I had higher performance by using a lemmatized version,  
as expected. However, lemmatization definitely took significantly longer, about 20 minutes to lemmatize the entire corpus on my MacBook Pro,    
versus about 5 minutes for stemming.

### 2. View Lemmatized Text
```python
#view sample of lemamtized text
lemmatized.sample(10)
```

|        | complaint_description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | assigned_division            |
|-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|
|  61231 | ['heat', 'build', 'say', 'people', 'bldg', 'may', 'provide', 'plugin', 'type', 'electrical', 'heater', 'apt']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Housing Inspection Services  |
|  58487 | ['sewage', 'seep', 'garage']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Plumbing Inspection Division |
|  49951 | ['severe', 'electricalplumbe', 'problem', 'fuse', 'blow', 'oftenlight', 'short', 'wind', 'blow', 'remodel', 'do', 'permit', 'post']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Housing Inspection Services  |
|  27925 | ['date', 'last', 'observe', 'aug', 'time', 'last', 'observe', 'floor', 'ground', 'exact', 'location', 'rear', 'bldg', 'build', 'type', 'residencedwelle', 'insectsrodent', 'damage', 'wall', 'dilapidate', 'structure', 'build', 'additional', 'information', 'trash', 'dump', 'illegally', 'front', 'house', 'backyard', 'overgrown', 'fill', 'trash', 'infest', 'rodent', 'retain', 'wall', 'fence', 'look', 'danger', 'collapse', 'plant', 'backyard', 'intrude', 'neighbor', 'property', 'affect', 'backyard', 'pave', 'look', 'site', 'history', 'similar', 'complaint', 'file', 'month', 'ago', 'look', 'like', 'action', 'take'] | Building Inspection Division |
| 122524 | ['proper', 'containment', 'remove', 'paint', 'item']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Housing Inspection Services  |
|  23362 | ['permit', 'application', 'file', 'status', 'owner', 'still', 'work', 'roofdeck', 'pywood', 'walkway', 'cut', 'roof', 'stairwell', 'deck', 'work', 'past', 'two', 'three', 'week']                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Building Inspection Division |
| 107161 | ['tear', 'exist', 'garage', 'excavate', 'large', 'anount', 'dirt', 'build', 'least', 'retain', 'wall', 'wout', 'permit']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Building Inspection Division |
| 107553 | ['power', 'washing', 'build', 'without', 'containment', 'debris', 'car', 'street']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Housing Inspection Services  |
| 121109 | ['vacant', 'build', 'register', 'wdbi', 'build', 'damage', 'fire', 'homeless', 'live', 'expire', 'build', 'permit', 'pa', 'ansol', 'system']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Building Inspection Division |
| 173027 | ['patron', 'call', 'state', 'year', 'problem', 'kitchen', 'sink', 'report', 'office', 'office', 'send', 'plumber', 'plumber', 'fix', 'anything', 'leave', 'never', 'come', 'back', 'tell', 'issue', 'bathroom', 'replace', 'tub', 'kitchen', 'sink', 'issue', 'fix', 'soon', 'run', 'water', 'horrible', 'smell', 'water', 'come', 'entire', 'unit', 'smell', 'horrible', 'dish', 'tub', 'almost', 'year', 'kid', 'take', 'anymore', 'today', 'leave', 'unit', 'go', 'homeless', 'shelter', 'kid', 'year', 'old', 'issue', 'get', 'point', 'cause', 'homeless', 'hey', 'would', 'like', 'help', 'asap']                                 | Housing Inspection Services  |


### 3. Check for Duplicates After Lemmatizing
After lemmatization we have reduced many words to the same form so it is necessary to check for duplicate entries afterwards. 
``` python
print(lemmatized.isna().sum())                          #print null values
print("Duplicated :", lemmatized.duplicated().sum())    #print duplicates
```
*complaint_description    0  
assigned_division        0  
dtype: int64
Duplicated : 13399* 

We can see that we have introduced 13,399 duplicates into the dataset which we will now remove:
``` python
lemmatized.drop_duplicates(inplace = True)
```

### 4. View Shape and Class Balance of Lemmatized Data
Now we will view the shape of our lemmatized data and the number of complaints in each class: 
``` python
print(lemmatized.shape) #print shape of lemmatized data
print(lemmatized.assigned_division.value_counts()) #print value counts for each division after lemmatization
```
*(160355, 2)
Housing Inspection Services       81811  
Building Inspection Division      61362  
Plumbing Inspection Division      10650  
Electrical Inspection Division     5444  
Disabled Access Division           1088  
Name: assigned_division, dtype: int64*     


### 5. Calculate Class Weights
Next we will calculate the class weights of each division, or class, in our dataset, we will pass these to our model which will    
adjust the amount of weight Keras places on each class when classifying. This forces the model to pay more attention to under   
represented classes by increasing their weights in the loss function. When they are missclassified the error will be weighted   
more heavily therefore penalizing miss-classifications of the smaller classes. 

We will calculate the class weights using Scikit-Learn's compute_class_weight function from the class_weight module. 

``` python
classes_lemmatized = lemmatized.assigned_division                           #select classes
lemmatized_class_weights = class_weight.compute_class_weight('balanced',    #compute balanced class weights
                                                 classes = np.unique(classes_lemmatized),y = classes_lemmatized)
lemmatized_class_weights = dict(enumerate(lemmatized_class_weights))        #convert class weights list to dictionary
```

Then we save the class weights so we can pass them to Keras later: 
``` python
#save class weights
with open(lemmatized_path + 'lemmatized_class_weights.pickle', 'wb') as f:
    pickle.dump(lemmatized_class_weights, f)
```


### 6. Split Features/Label and Binarize Label
Now we will split our features and label, and binarize our label so that each class is represented as a  
one-hot-encoded array. 

``` python
encoder = LabelBinarizer()                                          #define label binarizer

X_stemmed = stemmed.complaint_description                           #select stemmed complaints
y_stemmed = encoder.fit_transform(classes_stemmed)                  #select stemmed target
X_lemmatized = lemmatized.complaint_description                     # selected lemmatized complaints which will be our feature
y_lemmatized = encoder.fit_transform(lemmatized.assigned_division)  #fit label finarizer
class_names = encoder.classes_                                      #select and save class names to preserve order for later use

with open(lemmatized_path + 'class_names.pickle', 'wb') as f:       #save class names 
    pickle.dump(class_names, f)
```


### 7. Create Sequences using Tensorflow Tokenizer
Next we are going to create sequences to represent our text using Tensorflow's tokenizer. This will create sequences  
where each word is assigned a token id which is used to represent the word. We will pass in the lemmatized text prepared   
in the last step. Tensorflow's Tokenizer can could also accomplish many of the same operations we performed earlier,  
like lowercasing, but I  wanted to perform them myself for the experience. 

First we will fit the tokenizer on our text, the fit_on_texts method is used to update the internal vocabulary of the tokenizer,  
this is used to represent the words in the text and is where each word gets assigned a unique id to represent it. The more common a word is the lower its index.       

Then we call the texts_to_sequence method which is used to convert the tokens of a text corpus that are passed in into a sequence of integers that represent the words in the corpus. 

We determine how many total words to use for the tokenizer's vocabulary, we will use 10,000. Any words that are not in the 10,000   
most common words will be replaced by the out of vocabulary token. We will also set a max length for each sequence to 200, any sequences that are smaller will be padded so that inputs have uniform dimensions.  

``` python
#define tensorflow tokenizer for the lemmatized text
num_words = 10000   #number of words to use in vocabulary
lemmatized_tokenizer = Tokenizer(num_words=num_words, oov_token='<UNK>') #initialize the tokenizer with out of vocabulary token

#fit tokenizer to text
lemmatized_tokenizer.fit_on_texts(X_lemmatized)

#save the tokenizer so we can use it later to process data for predictions
with open(lemmatized_path + 'lemmatized_tokenizer.pickle', 'wb') as f:
    pickle.dump(lemmatized_tokenizer, f)


#define variables for word count and index
lemmatized_word_count = lemmatized_tokenizer.word_counts
lemmatized_word_index = lemmatized_tokenizer.word_index


#encode the data into a sequence 
X_lemmatized_sequences = lemmatized_tokenizer.texts_to_sequences(X_lemmatized)

#pad the sequences
X_lemmatized = pad_sequences(X_lemmatized_sequences, padding='post',
                truncating='post', maxlen=200)
```

Once the tokenizer is finished our data is ready to be fed to our model and each complaint is represented as a sequence of integers that correspond to a word's frequency in the corpus.

We can view the sequences now: 
``` python
display(X_lemmatized)
```
*array([[ 86, 151,   9, ...,   0,   0,   0],  
       [ 11,   5,   6, ...,   0,   0,   0],  
       [ 11,   5,   6, ...,   0,   0,   0],  
       ...,  
       [ 39,  55,  76, ...,   0,   0,   0],  
       [ 86, 173,  22, ...,   0,   0,   0],  
       [ 39, 436,  46, ...,   0,   0,   0]], dtype=int32)*  

Each integer in the sequences represents a word using its token id that was found during the fit_on_texts call. 



# 5. DistilBERT Model
The first model I implemented was a Long-Short Term Memory network, and provided a nice baseline before building the DistilBERT Model. 

I experimented with making the LSTM model more complex but with the fairly small size of our data the LSTM model performed best with an architecture that was relatively simple. However it struggled significantly with detecting the under represented classes, like the Disabled Services Division, which has only 1,000 complaints. If you are interested in that model it can be found in the LSTM_model notebook, here we will focus primarily on the DistilBERT model as it is the main focus of this project. 

DistilBERT is a condensed version of the popular BERT Transformer model. DistilBERT has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark. I experimented with using the full BERT model as well, but I found being able to quickly train and experiment was more valuable than the extra performance I could squeeze out of the full BERT.   

We will download the DistilBERT Transformer model and its tokenizer from the Hugging Face library. Hugging Face provides a transfer learning library consisting of many different models, tokenizers, and transformer. It allows us to easily harness the power of large and computationally expensive models by downloading a model with its pre-trained weights so that we can easily apply the complex models to our problem.

BERT makes use of an attention mechanism, or transformer, that learns contextual relations between words in a text. In their standard form, transformers includes two separate mechanisms — an encoder that is responsbile for reading input text and a decoder that produces a prediction for the task.  

In contrast to directional models a Transformer reads a sequence of words at once, which allows the Transformer to learn the context of the word based on its surroundings, and helps it to more accurately model text.  

We will download the DistilBERT tokenizer and DistilBERT Transformer model, and will use the DistilBERT model's encoder to create word embeddings for our text, we will then train a LSTM classification head on these embeddings, which achieves much better results than our standalone LSTM model. 

### 1. Create Train, Test, and Validation Sets
Our first step is to create train, test, and validation sets from the lemmatized data we prepared earlier.   
We will split the data using stratification so that each set has a balanced representation of classes.  

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)         #create train and test sets
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.5, random_state = 42, stratify = y_test) #split test into test and validation sets
```

### 2. Define DistilBERT Tokenizer
To use a BERT model the tokens that are fed into the model must be encoded using a BERT tokenizer so that the token IDs match up to the token IDs the BERT model was trained on. Because of this the tokenization approach used earlier, and that I used for my LSTM model, will not work for the DistilBERT model. 

Instead we will tokenize the text using the DistilBERT tokenizer that accompanies te DistilBERT model.

We will now define a tokenize function we can use with the DistilBERT Tokenizer that will accept two parameters: the sentences to tokenize and the Hugging Face tokenizer to use.  The function will tokenize our sentences and return two Tensorflow tensors, one will contain the token ID's after tokenization, and the other will contain masks that tell the model which tokens to pay attention to. Padding or out of vocabulary tokens will be masked out. 



``` python

MAX_LENGTH = 128
def tokenize(sentences, tokenizer):
    """
    Args:
        sentences: The sentences to tokenize.
        tokenizer: The Hugging Face tokenizer object to use.
    
    Returns:
        input_ids: The token id's from DistilBERT.
        input_masks: Which tokens to mask (ignore). 
    """
    input_ids, input_masks = [],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True,                
                                       max_length=MAX_LENGTH,truncation=True,padding='max_length',
                                       return_attention_mask=True,return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])      

    return (tf.convert_to_tensor(input_ids), tf.convert_to_tensor(input_masks))
```


### 3. Create DistilBERT Model
Next we are ready to create our DistilBERT model, we create the model using a config file, and set the dropout and attention dropout to 0.2, we also set output_hidden_states=True so that this layer outputs the raw hidden states, or embedding layers. We will train our LSTM classification head on these embeddings. We also freeze the DistilBERT layers as these layers are pre-trained and we want to build a classifier on top of them. 

During training we will measure the AUC and accuracy of our model on validation data, although accuracy is a poor metric in such an imbalanced dataset. We will evaluate the precision, recall and F1 Score of our model on test data after training. 
``` python
DISTILBERT_DROPOUT = 0.2
DISTILBERT_ATT_DROPOUT = 0.2
 
# Configure DistilBERT's initialization
config = DistilBertConfig(dropout=DISTILBERT_DROPOUT, 
                          attention_dropout=DISTILBERT_ATT_DROPOUT, 
                          output_hidden_states=True)
                          
#bare pre-trained DistilBERT model outputting raw hidden-states 
#needs head for classification
distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

# Make DistilBERT layers untrainable
for layer in distilbert.layers:
    layer.trainable = False
```

Now we are ready to build our model using the functional API:

``` python
LEARNING_RATE = 1e-3
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 64
NUM_STEPS = X_train_input_ids.shape[0] // BATCH_SIZE

def build_model(transformer, num_classes, max_length=MAX_LENGTH):
    
    """""""""
    Builds a BERT model for classification tasks using a Hugging Face 
    transformer with no head attached.
    
    Input:
      - transformer:  base Hugging Face transformer with no head.
      - max_length:   Controls the maximum number of encoded tokens in 
                      a sequence.
    
    Output:
      - model:        a compiled tf.keras.Model with added multi-class 
                      classification layerson top of the base Hugging Face 
                      transformer. 
    """""""""""
    
    #define metrics to monitor
    metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc'),
    ]

    # define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE) 
    
    # define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(MAX_LENGTH,), 
                                            name='input_ids', 
                                            dtype='int32')
    input_masks_layer = tf.keras.layers.Input(shape=(MAX_LENGTH,), 
                                                  name='input_attention', 
                                                  dtype='int32')

    

    # tf.tensor representing the hidden-state of the model's last layer
    embedding_layer = transformer([input_ids_layer, input_masks_layer])[0]
    # Bidirectional LSTM with return sequences True to output entire sequence of hidden states 
    X = Bidirectional(LSTM(MAX_LENGTH, return_sequences=True, dropout=0.2))(embedding_layer)
    # Global Max Pooling Layer
    X = GlobalMaxPool1D()(X)
    # Dense layer with ReLU activation
    X = Dense(MAX_LENGTH, activation='relu')(X)
    # dropout
    X = Dropout(0.2)(X)
    # output nodes using softmax activation for multi-class classification
    output = Dense(num_classes, 
                                   activation='softmax',
                                   kernel_initializer=weight_initializer,  
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(X)
    
    # Define the model
    model = tf.keras.Model([input_ids_layer, input_masks_layer], output)
    
    # Compile the model, Adam optemizer using categorical crossentropy
    model.compile(tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                  loss='categorical_crossentropy',
                  metrics=metrics)
    
    return model
```

Building Model:
``` python
model = build_model(transformer=distilbert,num_classes=y_train.shape[1])
```
Display Model Summary: 
``` python
model.summary()
```

<details>
<summary> Click to display model summary! </summary>

<p>

Model: "model"   
__________________________________________________________________________________________________   
 input_ids (InputLayer)         [(None, 128)]        0           []                                  
                                                                                                     
 input_attention (InputLayer)   [(None, 128)]        0           []                                  
                                                                                                     
 tf_distil_bert_model (TFDistil  TFBaseModelOutput(l  66362880   ['input_ids[0][0]',                 
 BertModel)                     ast_hidden_state=(N               'input_attention[0][0]']           
                                one, 128, 768),                                                      
                                 hidden_states=((No                                                  
                                ne, 128, 768),                                                       
                                 (None, 128, 768),                                                   
                                 (None, 128, 768),                                                   
                                 (None, 128, 768),                                                   
                                 (None, 128, 768),                                                   
                                 (None, 128, 768),                                                   
                                 (None, 128, 768)),                                                  
                                 attentions=None)                                                    
                                                                                                     
 bidirectional (Bidirectional)  (None, 128, 256)     918528      ['tf_distil_bert_model[0][7]']      
                                                                                                     
 global_max_pooling1d (GlobalMa  (None, 256)         0           ['bidirectional[0][0]']             
 xPooling1D)                                                                                         
                                                                                                     
 dense (Dense)                  (None, 128)          32896       ['global_max_pooling1d[0][0]']      
                                                                                                     
 dropout_19 (Dropout)           (None, 128)          0           ['dense[0][0]']                     
                                                                                                     
 dense_1 (Dense)                (None, 5)            645         ['dropout_19[0][0]']                
                                                                                                     
==================================================================================================   
Total params: 67,314,949   
Trainable params: 952,069   
Non-trainable params: 66,362,880   
__________________________________________________________________________________________________   

</p>
</details>

### 4. Creating Callbacks 

##### 1. Step Decay Schedule
First we will define a step decay schedule that will gradually accelerate the decay of our learning rate in addition to the acceleration provided by the Adam optimizer. 

``` python
#now we will define a step decay function to reduce our learning rate
def step_decay_schedule(initial_lr=LEARNING_RATE, decay_factor=0.75, step_size=10):
    """
    Wrapper function to create a LearningRateScheduler with a set step decay schedule
    """
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

#learning rate scheduler to decay the learning rate throughout training
lr_sched = step_decay_schedule(initial_lr=LEARNING_RATE, decay_factor=0.75, step_size=2)
```

##### 2. Model Checkpoint
Next we create a checkpoint to save our model when its performance improves on validation data. 
``` python
#define callbacks for our model
checkpoint = ModelCheckpoint(filepath = MODEL_PATH + f'BERT/model_{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_best_weights.h5', 
                             monitor='val_loss',
                             verbose=1,
                             save_weights_only = True, 
                             save_best_only=True,
                             mode='min')
```

##### 3. Early Stopping
We will implement early stopping if the model doesn't improve on validation data for 10 epochs. 
``` python
#implement early stopping if val loss doesn't improve for 10 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
```

##### 4. TQDM Progress Bar
Add a nice progress bar to help keep tracking of training progress.  
``` python
#tqdm progress bar
tqdm_callback = tfa.callbacks.TQDMProgressBar()
```



### 5. Train Model
Now we are ready to begin training our model. We will train our model for 100 epochs, unless early stopping stops training early, with a batch size of 64. We will also create a list of our callbacks to pass to our model.

``` python
EPOCHS = 100
BATCH_SIZE = 64
NUM_STEPS = X_train_input_ids.shape[0] // BATCH_SIZE

#define list of callbacks
callbacks = [checkpoint,
             early_stopping,
             lr_sched,
             tqdm_callback]

# train the model
history = model.fit(
    x = [X_train_input_ids, X_train_input_masks],
    y = y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    steps_per_epoch = NUM_STEPS,
    validation_data = ([X_val_input_ids, X_val_input_masks], y_val),
    verbose=2,
    callbacks=callbacks
)
```




### 6. Evaluate Model 
Now our model has finished training and we are ready to evaluate its performance on the test set. Our model reached the best validation AUC of 0.9839 on Epoch 8, with a validation loss of 0.3322. 

#### 1. Make Test Predictions
Evaluate model on test data:
``` python
# evaluate the model on the test data
y_pred = model.predict([X_test_input_ids, X_test_input_masks])
```

#### 2. Evaluate AUC on Test Data
Now we will evaluate the AUC of our model on test data. The AUC or Area Under Curve is a measurement of the model's ability to distinguish classes. It measures the area under the Receiver Operating Characteristics curve, which plots the True Positive rate, or recall on the Y-Axis against the False Positive rate on the X-Axis. 

``` python
weighted_auc = metrics.roc_auc_score(y_test, y_pred, 
                            average='weighted',multi_class='ovr')
print(f'Weighted AUC: {weighted_auc}')
```
*Weighted AUC: 0.9695763391216035*

#### 3. Evaluate Accuracy
Next we will evaluate the model's accuracy, as mentioned earlier accuracy is a poor metric, especially in such imbalanced data, as the model could achieve high accuracy by simply predicting the largest class. We will evaluate Precision, Recall, and the F1 next, as well as look at a Confusion Matrix, which are better measurements for imbalanced data, as is the AUC we measured previously.
``` python
#we take the argmax to get the prediction with highest probability
y_pred = np.argmax(y_pred, axis=1) 
accuracy = metrics.accuracy_score(y_true, y_pred, normalize=True)
print(f'Accuracy: {accuracy}')
```
*Accuracy: 0.8758561151079137*

#### 4. Confusion Matrix
First we will look at the class names from our encoder so we know what row in our Consufion Matrix corresponds to what class.
``` python
encoder.classes_
```
*array(['Building Inspection Division', 'Disabled Access Division',  
       'Electrical Inspection Division', 'Housing Inspection Services',    
       'Plumbing Inspection Division'], dtype='U30')*

Now we will create our confusion matrix:
``` python
#create confusion matrix of our test predictions
print(metrics.confusion_matrix(y_true,y_pred))
```
*[[5534   20   59  646  109]  
 [  26   65    1   20    0]  
 [  84    0  391   62   18]  
 [ 571   17   69 7910  144]  
 [ 142    1   21  147 1318]]*

 We can see that the model is doing very well at predicting the majority classes, and fairly well at predicting the 2nd most underrepresented class, the Electrical Inspection Division. Unfortunately its performance for the Disabled Access Division, the smallest class is still not great, it frequently mistook the Building Inspection Division and the Housing Inspection Division for this class. 

 #### 5. Create Classification Report
 Now we will create a classification report which shows the Precision, Recall, and F1 score for each class, as well as the average for each metric overall. 
 ``` python
 #create classification report
print(metrics.classification_report(y_true,y_pred))
```


              precision    recall  f1-score   support  
  
           0       0.87      0.87      0.87      6368  
           1       0.63      0.58      0.60       112  
           2       0.72      0.70      0.71       555  
           3       0.90      0.91      0.90      8711  
           4       0.83      0.81      0.82      1629  
    accuracy                           0.88     17375  
    macro avg      0.79      0.77      0.78     17375  
    weighted avg   0.88      0.88      0.88     17375  

Precision is equal to the True Positive / True Positive + False Positive, and answers the question "Of the predictions my model makes for a class, how many are correct?", or in other words how precise the predictions are.   

Recall is True Positive / True Positive + False Negative and instead answers the question "Out of all actual instances of this class how many did my model succesffuly identify correctly?" or how sensitive the model is to a class.   

F1 Score is the Harmonic Mean of Precision and Recall, and provides a balanced metric that shows how your model does with each metric taken into account.  

From these results we can conclude that the model excels at predicting the Building Inspection Division, the Housing Inspection Division, and  The Plumbing Inspection Division, it also performs fairly well on the Electrical Inspection Division, which is the second smallest class. On the Disabled Access Division the model still has some room for improvement, with an F1-Score of 0.60 and a Precision of 0.63 an recall of 0.58. I am going to explore feature augmentation steps like translating the corpus to another language and back to English to generate more data for the imbalanced classes next. 
