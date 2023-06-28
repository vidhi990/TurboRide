#!/usr/bin/env python
# coding: utf-8

# In[59]:


#build a model that predicts the total ride duration of taxi trips in New York City. 
#primary dataset is one released by the NYC Taxi and Limousine Commission,
#which includes pickup time, geo-coordinates, number of passengers, and several other variables.
#the other dataset is wearther report of new-york city


# In[1]:


pip install fastparquet


# In[15]:


##importing required modules
import pandas as pd
from fastparquet import ParquetFile


# In[18]:


#df = pd.read_parquet(r'/content/drive/MyDrive/DNN_PROJECT/yellow_tripdata_2020-01.parquet')
##importing the dataset
df = pd.read_parquet(r'C:\Users\vidhi shah\Downloads\yellow_tripdata_2020-01.parquet')
#df.head(10)
print("Yello_TripData_2020-01 \n",df.head(10))


# In[20]:



print("\n length of vendor id: ",len(df["VendorID"])) ##checking the length of the attribute


# In[21]:


##seperating date from datetime for further analysis
temp = []
for i in range(len(df["tpep_pickup_datetime"])):
  temp.append(str(df["tpep_pickup_datetime"][i]).split(" ")[0])
df["pickup_date"] = temp
print("\n dataset after seperating date and time: \n", df.head(2))


# In[22]:


df2 = pd.read_csv(r'C:\Users\vidhi shah\Downloads\\pymidterm\export.csv')
print("\n weather report: \n",df2.head(20))


# In[9]:


print("\n", df2.loc[0])
print(df.loc[0])


# In[23]:


res = df.merge(df2, how='inner', left_on=['pickup_date'], right_on=['date']) ##merging both the datasets


# In[24]:


##display the result
print("\n merged dataset: \n",res.head(5))


# In[25]:


##checking the length of the dataset
print("\n the length of the dataset:",len(res))


# In[26]:


##data pre-processing - to clean the data we will first check the null values
print("\n column wise count of null values: ")
res.isnull().sum() 


# In[27]:


## dimensionality reduction - removing unwanted fields that does not impact the target variable
res.drop("prcp", axis=1, inplace=True)


# In[28]:


res.drop("snow", axis=1, inplace=True)


# In[31]:


print("\n unique value present in wdir: ",res.wdir.unique()) ##checking the unique values to see if the attribute values will impact or not 


# In[32]:


res.drop(["wpgt",'tsun'], axis=1, inplace=True)


# In[33]:


res.drop('wdir', axis=1, inplace=True)


# In[34]:


import numpy as np
print("\n resultant dataset after removing all the unwanted columns: \n",res.head(15))


# In[35]:


print("\n data types of the columns: \n",df.dtypes) ##checking the data type of the columns


# In[36]:


res.drop('airport_fee', axis=1, inplace=True)
print("\n null check: \n",res.isnull().sum())


# In[37]:


##filling the null values with mean of the data which are numerical/continuous in nature  
res['passenger_count'] = res['passenger_count'].fillna(res['passenger_count'].mean())
res['RatecodeID'] = res['RatecodeID'].fillna(res['RatecodeID'].mean())
res['store_and_fwd_flag'] = res['store_and_fwd_flag'].fillna(0)
res['congestion_surcharge'] = res['congestion_surcharge'].fillna(res['congestion_surcharge'].mean())

print("\n result after cleaning the data: \n",res.isnull().sum())


# In[38]:


res['pres'] = res['pres'].fillna(0)
res['wspd'] = res['wspd'].fillna(res['wspd'].mean())
res['tavg'] = res['tavg'].fillna(res['tavg'].mean())
res['tmin'] = res['tmin'].fillna(res['tmin'].mean())
res['tmax'] = res['tmax'].fillna(res['tmax'].mean())

print("result after replacing null values with mean (data cleaning): \n",res.isnull().sum())


# In[43]:


##calculating the target field
res["trip_duration"] = (res['tpep_dropoff_datetime'] - res['tpep_pickup_datetime'])
res["trip_duration"] = res["trip_duration"].dt.seconds/60


# In[44]:


# Checking the size of dataset.
print("\n size of the dataset: ",res.shape)


# In[45]:


print("\n checking unique values of the columns: \n",res.nunique())


# In[46]:


print("\n data pre-processing : value counts of the columns are shown below for better understanding of the data: \n")
res.head()
print(res.VendorID.value_counts())
print(res.passenger_count .value_counts())
print(res.RatecodeID.value_counts())
print(res.store_and_fwd_flag.value_counts())
print(res.payment_type.value_counts())
print(res.mta_tax.value_counts())
print(res.improvement_surcharge.value_counts())
print(res.congestion_surcharge.value_counts())
print(res.trip_distance.value_counts())


# In[47]:


res.RatecodeID.value_counts()


# In[48]:


res.store_and_fwd_flag.value_counts()  #after conversion to numeric


# In[49]:


res.VendorID.value_counts()


# In[50]:


res.payment_type.value_counts()


# In[51]:


res = res.drop(res[(res.VendorID > 2)].index)


# In[52]:


res = res.drop(res[(res.payment_type == 0)].index)


# In[53]:


print("\n cleaned data: \n",res.head(5))


# In[54]:


res= res.drop(res[(res.trip_duration < 0) | (res.total_amount > 30)].index)


# In[55]:


res = res.drop(res[(res.tip_amount > 10) | (res.trip_distance > 10)].index)


# In[59]:


print("\n data info: \n")
print(res.info())


# In[61]:


print("correlation table: ",res.corr())


# In[63]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.heatmap(res.corr(),cmap="YlGnBu",annot=True,fmt=".1f")
print("\n correlation chart: ")
plt.show()


# In[65]:


print("\n scatter_plot to show the relationship between trip_duraion and trip_distance: ")
plt.scatter(res.trip_duration, res.trip_distance)
plt.xlabel('trip_duration')
plt.ylabel('trip_distance')
plt.show()


# In[66]:


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np


# In[67]:


print("\n columns of the data(headers): \n",res.columns)


# In[68]:


yellow_date_wise_sum = res.groupby(res['tpep_pickup_datetime'].dt.date).sum()[4:-8]


# In[69]:


##feature extraction
print("\n feature extraction using visualization: \n")
plt.plot(yellow_date_wise_sum.index,yellow_date_wise_sum['trip_duration'])
plt.legend(['Yellow'])
plt.gcf().autofmt_xdate()
plt.title('Daily Duration in Seconds')
plt.xlabel('Date')
plt.ylabel('Travel Duration (in secs)')


# In[70]:


yellow_PU_wise_sum = res[res['PULocationID'].notna()].groupby(res['PULocationID']).sum()


# In[71]:


plt.plot(yellow_PU_wise_sum.index[:-1],yellow_PU_wise_sum[:-1]['trip_duration'])


# In[73]:


res=res.loc[1:1000]
print("\n size of the data to be used for model implementation: ",len(res))


# In[74]:


res.drop("tpep_pickup_datetime", axis=1, inplace=True)


# In[75]:


res.drop("tpep_dropoff_datetime", axis=1, inplace=True)


# In[76]:


# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(res['store_and_fwd_flag'])

# printing label
print("\n labeling the store_and_fwd_flag: \n",label)


# In[77]:


# removing the column 'store_and_fwd_flag' from df
# as it is of no use now.
res.drop("store_and_fwd_flag", axis=1, inplace=True)
res["store_forward_ny"] = label


# In[78]:


print("\n final dataset:\n",res.head(3))


# In[82]:


##defining the target variable and predictors
predictor_cols=['VendorID','trip_distance',
       'passenger_count', 'RatecodeID','store_forward_ny',
       'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'congestion_surcharge', 'tavg',
       'tmin', 'tmax', 'wspd', 'pres']
print("\n predictor_cols: \n",predictor_cols)
target_col=['trip_duration']
print("\ntarget_col: \n",target_col)


# In[83]:


X=res[predictor_cols].values
y=res[target_col].values
 
### Sandardization of data ###
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)
 
# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)
 
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Quick sanity check with the shapes of Training and testing datasets
print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_test',X_test.shape)
print('y_test',y_test.shape)


# In[53]:


# Installing required libraries
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')

##applying dnn using tensorflow and keras
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import Dense


# In[56]:


# importing the libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5, input_dim=21, kernel_initializer='normal', activation='relu'))
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))

# Compiling the model
opt = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['mse', 'mae'])
# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 20, epochs = 100, verbose=1)


# In[57]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
history=model.fit(X_train,y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test,y_test),callbacks=[EarlyStopping(monitor='val_loss', patience=10)],shuffle=False)

def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show()
train_score = model.evaluate(X_train,y_train, verbose=0)
print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f ' 
% (np.sqrt(train_score[1]), train_score[2]))
test_score = model.evaluate(X_test,y_test, verbose=0)
print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f ' 
% (np.sqrt(test_score[1]), test_score[2]))
model_loss(history)


# In[58]:



# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 15, epochs = 5, verbose=0)

# Generating Predictions on testing data
Predictions=model.predict(X_test)

# Scaling the predicted trip duration data back to original duration scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)

# Scaling the y_test trip duration data back to original trip duration scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)

# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)

TestingData=pd.DataFrame(data=Test_Data, columns=predictor_cols)
TestingData['trip_duration']=y_test_orig
TestingData['Predictedtripduration']=Predictions
TestingData.head()

# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 15, epochs = 5, verbose=0)
 
# Generating Predictions on testing data
Predictions=model.predict(X_test)
 
# Scaling the predicted trip duration data back to original duration scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test trip_duration data back to original duration scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)
 
TestingData=pd.DataFrame(data=Test_Data, columns=predictor_cols)
TestingData['trip_duration']=y_test_orig
TestingData['Predictedtripduration']=Predictions
TestingData.head()


# In[ ]:


#conclusion : the predicted trip duration is as visible in 'Predictedtripduration' column and actual price is 'trip_duration'

