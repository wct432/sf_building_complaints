import config #config file contains API credentials from data.sfgov
import os
import pandas as pd 
from sodapy import Socrata

os.chdir('..')
parent_dir = os.getcwd()
data_path = parent_dir + '/data/'

client = Socrata("data.sfgov.org", config.app_key)

results = client.get("gm2e-bten", limit = 300000)

results_df = pd.DataFrame.from_records(results, index ='complaint_number')

results_df.to_csv(data_path + 'raw_data.csv')