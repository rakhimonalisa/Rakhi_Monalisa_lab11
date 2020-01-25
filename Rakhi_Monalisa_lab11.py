# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
path = "C:/Users/300997447/Desktop/New folder"
filename = 'wine.csv'
fullpath = os.path.join(path,filename)
data_mayy_wine = pd.read_csv(fullpath,sep=';')
# set the columns display and check the data
pd.set_option('display.max_columns',15)
print(data_mayy_wine.head())
print(data_mayy_wine.columns.values)
print(data_mayy_wine.shape)
print(data_mayy_wine.describe())
print(data_mayy_wine.dtypes) 
print(data_mayy_wine.head(5))
print(data_mayy_wine['quality'].unique())
pd.set_option('display.max_columns',15)
print(data_mayy_wine.groupby('quality').mean())

##############################

import matplotlib.pyplot as plt
plt.hist(data_mayy_wine['quality'])

#################################

import seaborn as sns
sns.distplot(data_mayy_wine['quality'])
# plot only the density function
sns.distplot(data_mayy_wine['quality'], rug=True, hist=False, color = 'r')
# Change the direction of the plot
sns.distplot(data_mayy_wine['quality'], rug=True, hist=False, vertical = True)
# Check all correlations
sns.pairplot(data_mayy_wine)
# Subset three column
x=data_mayy_wine[['fixed acidity','chlorides','pH']]
y=data_mayy_wine[['chlorides','pH']]
# check the correlations 
sns.pairplot(x)
# Generate heatmaps
sns.heatmap(data_mayy_wine[['fixed acidity']])
sns.heatmap(x)
sns.heatmap(x.corr())
sns.heatmap(x.corr(),annot=True)
##
import matplotlib.pyplot as plt
plt.figure(figsize=(10,9))
sns.heatmap(x.corr(),annot=True, cmap='coolwarm',linewidth=0.5)
##line two variables
plt.figure(figsize=(20,9))
sns.lineplot(data=y) 
sns.lineplot(data=y,x='chlorides',y='pH')
## line three variables
sns.lineplot(data=x)

######################################

data_mayy_wine_norm = (data_mayy_wine - data_mayy_wine.min()) / (data_mayy_wine.max() - data_mayy_wine.min())
data_mayy_wine_norm.head()

###################################

# check some plots after normalizing the data
x1=data_mayy_wine_norm[['fixed acidity','chlorides','pH']]
y1=data_mayy_wine_norm[['chlorides','pH']]
sns.lineplot(data=y1) 
sns.lineplot(data=x1)
sns.lineplot(data=y,x='chlorides',y='pH')

######################

from sklearn.cluster import KMeans
##from sklearn import datasets
model=KMeans(n_clusters=6)
model.fit(data_mayy_wine_norm)

########################

from sklearn.cluster import KMeans
##from sklearn import datasets
model=KMeans(n_clusters=3)
model.fit(data_mayy_wine_norm)

model.labels_
# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md=pd.Series(model.labels_)
data_mayy_wine_norm['clust']=md
data_mayy_wine_norm.head(10)
#find the final cluster's centroids for each cluster
model.cluster_centers_
#Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster.
#For an efficient cluster, the J-score should be as low as possible.
model.inertia_
#let us plot a histogram for the clusters
import matplotlib.pyplot as plt
plt.hist(data_mayy_wine_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
# plot a scatter 
plt.scatter(data_mayy_wine_norm['clust'],data_mayy_wine_norm['pH'])
plt.scatter(data_mayy_wine_norm['clust'],data_mayy_wine_norm['chlorides'])


######################################

from sklearn.cluster import KMeans
#from sklearn import datasets
model=KMeans(n_clusters=10)
model.fit(data_mayy_wine_norm)

model.labels_
# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md=pd.Series(model.labels_)
data_mayy_wine_norm['clust']=md
data_mayy_wine_norm.head(10)
#find the final cluster's centroids for each cluster
model.cluster_centers_
#Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster.
#For an efficient cluster, the J-score should be as low as possible.

##print(model.inertia_)

#let us plot a histogram for the clusters
import matplotlib.pyplot as plt
plt.hist(data_mayy_wine_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
# plot a scatter 
plt.scatter(data_mayy_wine_norm['clust'],data_mayy_wine_norm['pH'])
plt.scatter(data_mayy_wine_norm['clust'],data_mayy_wine_norm['chlorides'])

#####################################

from sklearn.cluster import KMeans
#from sklearn import datasets
model=KMeans(n_clusters=5)
model.fit(data_mayy_wine_norm)

model.labels_
# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md=pd.Series(model.labels_)
data_mayy_wine_norm['clust']=md
data_mayy_wine_norm.head(10)
#find the final cluster's centroids for each cluster
model.cluster_centers_
#Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster.
#For an efficient cluster, the J-score should be as low as possible.
model.inertia_
#let us plot a histogram for the clusters
import matplotlib.pyplot as plt
plt.hist(data_mayy_wine_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
# plot a scatter 
plt.scatter(data_mayy_wine_norm['clust'],data_mayy_wine_norm['pH'])
plt.scatter(data_mayy_wine_norm['clust'],data_mayy_wine_norm['chlorides'])