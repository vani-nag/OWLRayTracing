'''
#Single version
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
#df = pd.read_csv('/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/porto.csv',usecols=[0,1,2],header=None, names=['x','y','z'],nrows=100000)
df = pd.read_csv('/home/min/a/nagara16/fast-cuda-gpu-dbscan/CUDA_DCLUST_datasets/3D_iono.txt',usecols=[0,1,2],header=None, names=['x','y','z'],nrows=100000)
X = np.array(df)
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
df = pd.DataFrame(distances.flatten())
maxValues = df.max()
print("90th percentile: ",np.percentile(distances,90))
print("Max: ",max(maxValues))
#df.to_csv("knn_porto.csv")
#df.describe().to_csv("knn_3droad_dist_summary.csv")
'''


'''
Used to find max, 99th and 90th percentile distance between any two points in the dataset.
'''
#Multi-run
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
#df = pd.read_csv('/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/porto.csv',usecols=[0,1,2],header=None, names=['x','y','z'],nrows=100000)

#file_names = ["/home/min/a/nagara16/fast-cuda-gpu-dbscan/CUDA_DCLUST_datasets/3D_iono.txt", "/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/porto.csv"]
file_names = ["/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/TrueKNN/kitti.csv"]
sizes = [100000, 200000, 400000, 800000]

#Number of neighbors (k)
neighs = {}
neighs[100000] = [5, 316]
neighs[200000] = [5, 447]
neighs[400000] = [5, 660]
neighs[800000] = [5, 894]

'''neighs = {}
neighs[100000] = 5
neighs[200000] = 5
neighs[400000] = 5
neighs[800000] = 5'''

#Redirect output to a file
for file_name in file_names:
	print("-----------------------------"+file_name+"------------------------------")
	for size in sizes:
		print("Size = ",size)	
		df = pd.read_csv(file_name,sep=' ',usecols=[0,1,2],header=None, names=['x','y','z'],nrows=size)
		X = np.array(df)
		nbrs = NearestNeighbors(n_neighbors=(neighs[size]+1), algorithm='ball_tree').fit(X)
		distances, indices = nbrs.kneighbors(X)
		df = pd.DataFrame(distances)
		maxValues = df.max()
		print("90th percentile: ",np.percentile(distances,90))
		print("99th percentile: ",np.percentile(distances,99))
		print("Max: ",max(maxValues))

