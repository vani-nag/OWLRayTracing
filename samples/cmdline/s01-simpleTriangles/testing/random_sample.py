import pandas as pd
import time

df = pd.read_csv("/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/porto.csv",nrows=1000000)
#df = pd.read_csv("/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/3D_uniform_dist.csv",nrows=400000)
list1 = df.values.tolist()

#Choose random samples
from random import sample

start = time.time()
samples = sample(list1,100)
sample_time = time.time()-start
print("Sampling time: "+str(sample_time))
#print("Random sample of 10", samples)

from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array(samples)

#Run KNN
start = time.time()
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
knn_time = time.time()-start
print("KNN time: "+str(knn_time))

#Find min distance
start = time.time()
distances, indices = nbrs.kneighbors(X)
min1 = min(distances, key=lambda x: x[1])[1]
min2 = min(distances, key=lambda x: x[2])[2]
min3 = min(distances, key=lambda x: x[3])[3]
print("Min distance = "+str(min(min1,min2,min3)))
#min_radius = min(distances.all())
print("Total time: "+str(time.time()-start+knn_time+sample_time))

df = pd.DataFrame(distances)
df.to_csv("random_sample_knn_porto.csv")
df.describe().to_csv("random_sample_knn_porto_dist_summary.csv")

