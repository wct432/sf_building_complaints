- [1. Introduction](#1-introduction)
# 1. Introduction
The goal of this project is to build a Natural Language Processing classifier using Tensorflow, Keras, Hugging Face, and other libraries  
that the City of San Francisco could use to classify complaints received by the building department so the complaints can be forwarded to   
the proper division, such as the Electrical Services Divison or the Building Inspection Division.  

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
