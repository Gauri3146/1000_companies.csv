#!pip install matplotlib
#step1: importing all the package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#step2
data=pd.read_csv('1000_Companies.csv')
#############
data.head()
data.tail()
data.info()
data.describe()

data.columns
data.isna().sum()
########################################################
sns.heatmap(data.corr(),annot=True)
##################################
plt.scatter(x=data['R&D Spend'],y=data['Profit'])
sns.scatterplot(x=data['Administration'],y=data['Profit'])

sns.scatterplot(x=data['Marketing Spend'])

sns.barplot(x=data['State'],y=data['Profit'])
sns.boxplot(x=data['State'],y=data['Profit'])
##########################################

data['State'].unique()
data['State'].value_counts()

#########################################
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

data['State']=encoder.fif_transform(data['State'])
#############################################
#seggregate input output
x=data.drop(["Profit"],axis=1)
y=data["Profit"]
       #############################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)
 ############################33
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

regressor.coef_
regressor.intercept_









