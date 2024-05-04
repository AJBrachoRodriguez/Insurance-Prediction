#!/usr/bin/env python
# coding: utf-8

# # **Project "*Medical Insurance Cost Prediction*"**

# In[ ]:


Image('/Users/alexangelbracho/Desktop/projectAWS/images/imageInsurance.jpeg') 


# # **1. Overview**
# 

# 
# 
# 
# 
# ### The medical insurance is one of the most important issues of any society. Therefore, the costs associated with this is very critical in any familiar budget. In this project, we´ll be dealing with two main questions:
# 
# ### **--->** What are the primary factors influencing medical insurance expenses?
# 
# ### **--->** How accurate is the Linear Regression machine learning technique in predicting medical expenses?

# ## *Core Question*
# 
# ### **What is the estimated cost of a Medical Insurance?**
# 

# # **2. Preprocessing**

# In[3]:


# data and array managament
import pandas as pd
import numpy as np

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sb
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')

# sckikit-learn libraries
import sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle

# spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import SQLContext
from boto3.session import Session


# ## **2.1 Data Cleaning**
# 
# ### We´ll use Spark to make this section.
# 

# In[4]:


# Create the Spark Session

spark = SparkSession.builder.appName('sparkInsurance').getOrCreate()

spark


# In[97]:


# Create the table 

insurance_df = spark.read.csv('./file_csv/medical_insurance.csv', sep=',', header=True)

insurance_df.show(5)


print(insurance_df.count())


# In[98]:


insurance_df.count()


# In[99]:


## Drop Duplicates

insurance_df = insurance_df.dropDuplicates()


# In[101]:


insurance_df.count()


# In[102]:


## Drop Null values

insurance_df = insurance_df.na.drop(how='any')


# In[103]:


insurance_df.count()


# ### There were no null values in the dataset.

# In[104]:


# rename the column 'bmi' by 'body_mass_index'

insurance_df = insurance_df.withColumnsRenamed({'bmi':'body_mass_index'})


# In[105]:


insurance_df.show(3)


# In[106]:


# here, we have the charges in ascending order 
insurance_df.sort('charges')[['charges']].show()


# In[107]:


## 5. Apply filters :
## charges & smoker

insurance_df_filter = insurance_df.filter((insurance_df.charges > 35000) & (insurance_df.smoker == 'yes' ) )
insurance_df_filter.show()
print(f'The number of people who smoke and spend more than $35.000:',insurance_df_filter.count())


# In[108]:


insurance_df_filter = insurance_df.filter((insurance_df.charges > 35000) & (insurance_df.smoker == 'no' ) )
insurance_df_filter.show()
print(f'The number of people who don´t smoke and spend more than $35.000:',insurance_df_filter.count())


# ### From the previous data, we can see that there are 274 people who are smokers paying more than $35.000 in insurance versus 6 not being.

# In[109]:


## 6. Number of classes per feature
## smoker

insurance_df[['smoker']].distinct().show()
print('The number of classes here is:',insurance_df[['smoker']].distinct().count())


# In[110]:


## region

insurance_df[['region']].distinct().show()
print('The number of classes here is:',insurance_df[['region']].distinct().count())


# In[111]:


## age

insurance_df[['age']].distinct().show()
print('The number of classes here is:',insurance_df[['age']].distinct().count())


# In[112]:


## children

insurance_df[['children']].distinct().show()
print('The number of classes here is:',insurance_df[['children']].distinct().count())


# In[113]:


## sex

insurance_df[['sex']].distinct().show()
print('The number of classes here is:',insurance_df[['sex']].distinct().count())


# In[114]:


## smoker

insurance_df[['smoker']].distinct().show()
print('The number of classes here is:',insurance_df[['smoker']].distinct().count())


# In[115]:


## charges

insurance_df[['charges']].distinct().show()
print('The number of classes here is:',insurance_df[['charges']].distinct().count())


# In[116]:


insurance_df.show(5)


# In[117]:


insurance_df.count()


# In[118]:


#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

insurance_df_pd = insurance_df.toPandas()


# In[119]:


insurance_df_pd.count()


# ## **2.2 Data Coding**

# #### In this part, we´ll perform some ***data coding*** from the file preprocessed in the previous step with Spark.

# In[120]:


insurance_df_pd.head(10)


# ### Now, we´ll use the describe method to check some properties of the dataset.

# In[121]:


insurance_df_pd.describe()


# ### And then, some general info about the dataset. 

# In[122]:


insurance_df_pd.info()


# ### We can see there are three non-numerical variables: *sex,smoker and region*. We´ll "code" them. We´ll use the method of Scikit-learn named *LabelEncoder()*.

# In[126]:


# sex

encoder = preprocessing.LabelEncoder()

insurance_df_pd['sex']= encoder.fit_transform(insurance_df_pd['sex'])

insurance_df_pd['sex']


# In[127]:


# smoker

encoder = preprocessing.LabelEncoder()

insurance_df_pd['smoker']= encoder.fit_transform(insurance_df_pd['smoker'])

insurance_df_pd['smoker']


# In[128]:


# region

encoder = preprocessing.LabelEncoder()

insurance_df_pd['region']= encoder.fit_transform(insurance_df_pd['region'])

insurance_df_pd['region']


# #### Now, we´ve done the *data coding* we can take a glance at how the dataset looks like right now.

# In[129]:


insurance_df_pd.head(5)


# ### We can see there are no categorical (non-numerical) values. Now, let´s take a look at the other library: Seaborn. We need to do further analysis with the heat map and the Pearson´s correlation matrix with all the variables to establish some relation between them (if there is some).

# # **3. EDA (Exploratory Data Analysis)**

# ## **3.1 Visualization**

# In[156]:


insurance_df_pd.head(2)


# In[157]:


sb.pairplot(insurance_df_pd[['age', 'sex', 'body_mass_index', 'children', 'smoker', 'region', 'charges']], diag_kind='kde')


# In[158]:


sb.pairplot(insurance_df_pd[['age', 'sex', 'body_mass_index', 'children', 'smoker', 'region', 'charges']], hue='smoker')


# ### We can see that the sex feature is balanced as well the region. On the other hand, we´ll find out how is the correlation between the variables in the dataset. Firstly, let´s see the *heatmap*.

# In[155]:


# heatmap

sb.heatmap(insurance_df_pd[['age', 'sex', 'body_mass_index', 'children', 'smoker', 'region', 'charges']].corr(),annot = True)


# ### From the Perason´s correlation we can see that the variables that are more related are "smoker" (feature) and "charges" (labels) with a number of 0,79 (the maximum possible value is 1). Moreover, we can see the correlation of all the variables with the target ('charges').

# In[131]:


corr_matrix = insurance_df_pd.corr()
corr_matrix


# #### Moreover, we can see the correlation of all the variables with the label ('charges').

# In[132]:


corr_matrix['charges'].sort_values(ascending=False)


# #### Once again, we see the closest relation is between smoker and charges. This will be useful for the selection of the attributes.

# ## **3.2 Selection of the attributes**

# In[133]:


X = insurance_df_pd['smoker']   ## X

Y = insurance_df_pd['charges']   ## Y


# In[134]:


X.shape


# In[135]:


Y.shape


# In[136]:


X_smoker = np.array(X).reshape(-1,1)
Y_charges = np.array(Y).reshape(-1,1)

print(X_smoker.shape)
print(Y_charges.shape)


# ##  **3.3 Train/Test/Val split**

# In[137]:


X_train,X_val, Y_train, Y_val = train_test_split(X_smoker,Y_charges, test_size=0.2)


# In[138]:


print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)


# ## **3.4 Scaler**

# In[139]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)


# In[140]:


X_train


# In[141]:


Y_train


# In[142]:


X_val = scaler.transform(X_val)


# # **4. Training of the model**
# 
# ### We´ll use the *Linear Regression* algorithm because we need to find a continous values, that is, the cost of a medical insurance.

# ## **4.1 Training the model**

# In[143]:


# here, we fix the model as Linear Regression

model = sklearn.linear_model.LinearRegression()
model.fit(X_train,Y_train)


# ## **4.2 Predictions of the model**

# In[166]:


print(X_val.shape)
Y_pred = model.predict(X_val)
print(Y_pred.shape)
print("Here we show the first 10 predictions:")
print(Y_pred[:10])


# In[145]:


list_of_predictions = pd.DataFrame(Y_pred)


# In[146]:


list_of_predictions.head(5)


# ## **4.3 Evaluation of the model**

# ### In the *Linear Regression* machine learning algorithm, we use *metrics* to calculate an error to summarize the predictive skill of a model. There are several sort of errors we could use, for instance,
# 
# **---->** *Mean Squared Error*
# 
# **---->** R^2 coefficient
# 
# **---->** *Root Mean Squared Error*
# 
# **---->** *Mean Absolute Error*
# 
# ### Among them, we´ll use the *Mean Square Error* because of the fact that this one is less suitable to outliers. 

# ### **4.3.1** ***Mean Squared Error***

# In[147]:


# Mean Abosulte Error (MAE)
# here, we´ll use the library of SciKit-learn

errors = sklearn.metrics.mean_absolute_error(Y_val,Y_pred)


# In[148]:


print('The standard deviation is: $',errors)


# #### It means that there might be an error or difference of $6069 when the prediction is done. 

# ### **4.3.2** ***R^2 coefficient***

# In[149]:


# R2 coefficient

r2_coefficient = r2_score(Y_val,Y_pred)
print('The R^2 coefficient is:',r2_coefficient)


# #### It means that the models will predict correctly the cost of the medical insurance 60 times every 100 persons. 

# # **5. Results**

# ### ---> The cost of a medical insurance being smoker is $ 32.283 and $8.468 not being. 
# ### ---> The main factor that influences the medical insurance costs is the feature “Smoker”.  
# ### ---> 3. In this case, the Linear Regression algorithm is 65%  accurate with $5659 of error in the predictions of the costs of the medical insurance.

# # **6. Storage**

# ## **6.1 Model**

# In[159]:


def saveModel(name_model):
  #name = name_model
  model_file = f"{name_model}" + ".pkl"

  with open(model_file,'wb') as f:
      pickle.dump(name_model,f)


# In[160]:


saveModel(model)


# ## **6.2 Scaler**

# In[161]:


def saveScaler(name_scaler):

  scaler_file = f'{name_scaler}'+'.pkl'

  with open(scaler_file,'wb') as f:
      pickle.dump(scaler,f)


# In[162]:


saveScaler(scaler)

