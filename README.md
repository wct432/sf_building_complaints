- [1. Introduction](#1-introduction)
- [2. Exploration and Visualizations](#2-exploration-and-visualizations)
    - [1. Fetch and Load Data](#1-fetch-and-load-data)
    - [2. Explore Data Demographics](#2-explore-data-demographics)
# 1. Introduction
The goal of this project is to build a Natural Language Processing multi-class Classifier using Tensorflow, Keras, Hugging Face,   
and other libraries that the City of San Francisco could use to classify complaints received by the building department so the   
complaints could be forwarded to the proper division, such as the Electrical Services Divison or the Building Inspection Division.  

I found the dataset on https://datasf.org/ which is a collection of public datasets the City of San Francisco has made available.   
The dataset consists of 184,593 different complaints that the Building Department has received and include other information   
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

# 2. Exploration and Visualizations
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

### 2. Explore Data Demographics
Now we are ready to explore our data, we want to get a feel for the dataframe and for the distribution of our data amongst the  
different classes.  

View Head of Data:
``` python
df.head()
```
|    | complaint_description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | assigned_division            |
|---:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|
|  0 | Caller reporting that the water is discolored. orange color in the sink, water heater(stains) drizzles down the walls;  the exterior windows were painted, the people that were painting wore mask, caller states they were not informed of the particles that were coming in the window. the building was tested for asbestos and the finding were not disclosed to the tenants. caller states the management is doing there own testing, caller states there was a gas leak and they were not given the reason why pgand e was turned off. if elevator is out for a few weeks it makes it hard on the elders in the building to get out. | Housing Inspection Services  |
|  1 | Date last observed: 03-oct-17;    time last observed: 10/6/17--noon;    exact location: rear bldg;    building type: residence/dwelling   illegal unit; work w/o permit; ;    additional information: appears a room is being added to an existing shed.;                                                                                                                                                                                                                                                                                                                                                                                  | Building Inspection Division |
|  2 | Date last observed: 08-dec-17;    exact location: main bldg;    building type: residence/dwelling   illegal unit; ;    additional information: anonymous complaint of illegal units in building;                                                                                                                                                                                                                                                                                                                                                                                                                                           | Housing Inspection Services  |
|  3 | Date last observed: 19-aug-17;    time last observed: shapiro;    exact location: main bldg;    building type: commercial/business   work beyond scope of permit; ;    additional information: again i'm awakened by the sounds of heavy equipment rolling into this site.  4:50am on a saturday.  yesterday was 4:30am.  this can not be legal.;                                                                                                                                                                                                                                                                                          | Building Inspection Division |
|  4 | Date last observed: 14-aug-17;    time last observed: 8 am;    floor: n/a;    unit: n/a;    exact location: main bldg;    building type: residence/dwelling   other building; ;    additional information: contractor is occupying street and storing equipment and dumpsters along entire street front of project and no valid street use permit is displayed - only an expired one;                                                                                                                                                                                                                                                      | Building Inspection Division |
