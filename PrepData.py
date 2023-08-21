import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import random 
from sklearn.utils import shuffle

#Seed and preparation
random.seed(999)
np.random.seed(999)

#Path
path = os.path.dirname(os.path.abspath(__file__))
folder=['\\combined_data\\']
combined_data=[]
station_name=[]


#Read, create metadata and save in prep_data folder
for file_name in glob.glob(path+folder[0]+'*.csv'):
    
    #Read and fill missing values
    x = pd.read_csv(file_name, low_memory=False,parse_dates=["datetime"],index_col=0)
    x["datetime"]=pd.to_datetime(x["datetime"])
    x=x.set_index('datetime')
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x=x.interpolate(method='ffill')

    ## Get variance metadata ####
    var=x.iloc[:,:3].var()
    for i in range(3):
        x.loc[:, 'var'+str(i)] = var.iloc[i]
    ### end get variance ####

    ## Get sem metadata ####
    sem=x.iloc[:,:3].sem()
    for i in range(3):
        x.loc[:, 'sem'+str(i)] = sem.iloc[i]
    ### end get sem ####


    ## Get min-max capacity physical metadata ####
    mini=x.iloc[:,3:5].min()
    maxi=x.iloc[:,3:5].max()
    for i in range(2):
        x.loc[:, 'min'+str(i)] = mini.iloc[i]
    for i in range(2):
        x.loc[:, 'max'+str(i)] = maxi.iloc[i]    
    ### end get sem ####    
    
    combined_data.append(x)
    sn=os.path.basename(file_name)
    station_name.append(sn[:len(sn)-4])
    x.to_csv(path+"\\prep_data\\"+sn, index=True)
