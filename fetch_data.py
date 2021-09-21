import config #config file contains API credentials
import os
import pandas as pd 
from sodapy import Socrata

working_dir = os.getcwd()
path = working_dir + '/data/'

client = Socrata("data.sfgov.org", config.app_key)

results = client.get("gm2e-bten", limit = 300000)

results_df = pd.DataFrame.from_records(results, index ='complaint_number')

results_df.to_csv(path + 'complaints.csv')