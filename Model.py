import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler,MaxAbsScaler,RobustScaler, Normalizer, normalize
from sklearn.neighbors import KNeighborsRegressor
import random 
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator

#Seed, path, etc,
random.seed(999)
np.random.seed(999)

path = os.path.dirname(os.path.abspath(__file__))
folder=['\\prep_data\\']
combined_data=[]
station_name=[]

#Read prepared data
for file_name in glob.glob(path+folder[0]+'*.csv'):
    x = pd.read_csv(file_name, low_memory=False,parse_dates=["datetime"])
    x["datetime"]=pd.to_datetime(x["datetime"])
    x=x.set_index('datetime')
    combined_data.append(x)
    sn=os.path.basename(file_name)
    station_name.append(sn[:len(sn)-4])



#DAZLS algorithm
class DAZLS(BaseEstimator):
    def __init__(self):
        self.__name__="DAZLS"
        self.xscaler=None
        self.x2scaler=None
        self.yscaler=None
        self.model1=None
        self.model2=None
        self.mini=None
        self.maxi=None
        self.on_off=None
        self.y_pred_test=None
        self.y_test=None
    def fit(self, data, xindex,x2index,yindex,n,clf,clf2,n_delay,cc):
        x_index=list(set(np.arange(0,nn))-set([n]))
        y_index=n
        on_off=np.asarray(combined_data[n].iloc[:,[n_delay*3+4,n_delay*3+5]])
        ###### GPS DIFFERENCE#######
        diff_index1=[n_delay*3+2,n_delay*3+3] #GPS location
        diff_index2=list(np.arange(n_delay*3+8,cc)) # Variance and SEM
                                     
        for nx in range(nn):
            for ff in diff_index1:
                combined_data[nx].iloc[:,ff]=(combined_data[nx].iloc[:,ff]-combined_data[n].iloc[:,ff])
            for fff in diff_index2:
                combined_data[nx].iloc[:,fff]=(combined_data[nx].iloc[:,fff]-combined_data[n].iloc[:,fff])
        ####################  CALIBRATION #################################
        temp_data=[combined_data[ind] for ind in x_index]#Without the target substation
        ori_data=np.concatenate(temp_data,axis=0)
        test_data=np.asarray(combined_data[y_index])
        X,X2, y= ori_data[:,xindex],ori_data[:,x2index],ori_data[:,yindex]
        X_train,X2_train, y_train = shuffle(X,X2, y, random_state=999) #just shuffling
        X_test,X2_test,y_test=test_data[:,xindex],test_data[:,x2index],test_data[:,yindex]
        xscaler=MinMaxScaler(clip=True)
        x2scaler=MinMaxScaler(clip=True)
        yscaler=MinMaxScaler(clip=True)
        X_scaler=xscaler.fit(X_train)
        X2_scaler=x2scaler.fit(X2_train)
        y_scaler=yscaler.fit(y_train)
        X_train=X_scaler.transform(X_train)
        X_test=X_scaler.transform(X_test)
        X2_train=X2_scaler.transform(X2_train)
        X2_test=X2_scaler.transform(X2_test)
        y_train=y_scaler.transform(y_train)
        y_test=y_scaler.transform(y_test)*on_off
        ###### MIN MAX CAPACITY ######
        mini=np.asarray(test_data[:,[-4,-3]])[-1]
        maxi=np.asarray(test_data[:,[-2,-1]])[-1]
        mini=y_scaler.transform(mini.reshape(1,-1))[0]*on_off
        maxi=y_scaler.transform(maxi.reshape(1,-1))[0]*on_off
        #####################
        ####################
        clf.fit(X_train,y_train)
        y_pred_train=clf.predict(X_train)
        X2_train=np.concatenate((X2_train,y_pred_train),axis=1)
        clf2.fit(X2_train,y_train)
        self.xscaler=X_scaler
        self.x2scaler=X2_scaler
        self.yscaler=y_scaler
        self.model1=clf
        self.model2=clf2
        self.mini=mini
        self.maxi=maxi
        self.on_off=on_off
        y_pred_test=clf2.predict(np.concatenate([X2_test,clf.predict(X_test)],axis=1))*on_off
        self.y_pred_test=(y_pred_test-np.min(y_pred_test,axis=0))/(np.max(y_pred_test,axis=0)-np.min(y_pred_test,axis=0)+0.0000000000001)*(maxi-mini)+mini
        self.y_test=y_test
    def predict(self):
        return self.y_pred_test

    def score(self,verbose=True):
        RMSE=(mean_squared_error(self.y_test,self.y_pred_test))**0.5
        R2=r2_score(self.y_test,self.y_pred_test)
        if verbose:
            print('RMSE test=',RMSE)
            print('R-sqr=',R2)
        return RMSE, R2


###Create delay/lags (NOT USED HERE, BUT MAY BE USEFUL)
n_delay=1
for i in range(len(combined_data)):
    delaylist=[]
    if n_delay>1:
        for n in range(1,n_delay):
            delay=combined_data[i].iloc[:,:3].shift(-n)
            delay.columns=['delay'+str(n)+'_'+combined_data[i].columns[0],'delay'+str(n)+'_'+combined_data[i].columns[1],'delay'+str(n)+'_'+combined_data[i].columns[2]]
            delaylist.append(delay)
    combined_data[i]=pd.concat([*delaylist,combined_data[i].iloc[:,:]],axis=1).dropna()



# CHOOSE THE DATA, METADATA and TARGET, ETC. BY INDEX
cc=len(combined_data[0].columns)-4
xindex=list(np.arange(0,n_delay*3))+list(np.arange(n_delay*3+2,cc))
x2index=list(np.arange(n_delay*3+2,cc))
yindex=[n_delay*3,n_delay*3+1]


#PREPARATION
ori_combined_data=combined_data.copy() #Good procedure to prevent data changing in-place
clf=KNeighborsRegressor(n_neighbors=20,weights='uniform') #any model can be specified, this is the domain model
clf2=KNeighborsRegressor(n_neighbors=20,weights='uniform') #any model can be specified, this is the adaptation model

nn=len(station_name)
for n in range(nn): #loop through all stations (leave one out)
    print(station_name[n])
    model=DAZLS() #Initialize DAZLS model
    model.fit(data=ori_combined_data, xindex=xindex,x2index=x2index,yindex=yindex,n=n,clf=clf,clf2=clf2,n_delay=n_delay,cc=cc) #Fit model
    y=model.predict() #get predicted y
    model.score() #print prediction performance