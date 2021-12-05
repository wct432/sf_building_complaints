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
