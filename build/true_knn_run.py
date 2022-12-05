import os
       
#Varying start radius
#radii_3droad = [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08]
#radii = [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1]
#radii_porto = [0.000009, 0.00003, 0.00011, 0.00012, 0.00014, 0.00025, 0.00026, 0.0003, 0.026]

#We test TrueKNN with k=5 and k=sqrt(DatasetSize). knn_datasize is a dictonary with DatasetSize as key
knn_datasize = {}
knn_datasize[100000] = [5,316]
knn_datasize[200000] = [5,447]
knn_datasize[400000] = [5,660]
knn_datasize[800000] = [5,894]

'''knn_datasize[100000] = [316]
knn_datasize[200000] = [447]
knn_datasize[400000] = [660]
knn_datasize[800000] = [894]'''

'''
radii_<Dataset> is a dictonary holding the start radius for trueKNN
This needs to be filled in using output of owl/samples/cmdline/s01-simpleTriangles/testing/random_sample.py
'''
#Porto
radii_porto = {}
radii_porto[100000] = [0.8]
radii_porto[200000] = [0.5]
radii_porto[800000] = [0.8]
'''
radii_porto[100000] = [0.000009, 0.00007, 0.0001, 0.0002, 0.0004, 0.0005, 0.022]
radii_porto[200000] = [0.00002, 0.00003 ,0.00006, 0.0001, 0.0002, 0.022]
radii_porto[800000] = [0.00001, 0.00002, 0.00006, 0.0001, 0.0003, 0.0299]
'''

#3DIono
radii_3diono = {}
'''radii_3diono[100000] = [0.072, 0.086, 0.18, 0.2, 0.4, 22.33]
radii_3diono[200000] = [0.14, 0.35, 0.39, 0.48, 0.5, 137.77]
radii_3diono[400000] = [0.2, 0.24, 0.35, 0.6, 0.7, 136.95]
radii_3diono[800000] = [0.19, 0.37, 0.47, 0.56, 0.9, 127.1]'''
radii_3diono[100000] = [0.072]
radii_3diono[200000] = [0.14]
radii_3diono[400000] = [0.2]
radii_3diono[800000] = [0.19]

#Kitti
radii_kitti = {}
'''radii_kitti[100000] = [0.04, 0.08, 0.1, 0.11, 0.15, 16.3]
radii_kitti[200000] = [0.08, 0.13, 0.07, 0.28, 0.15, 15.8]
radii_kitti[400000] = [0.03, 0.07, 0.1, 0.2, 0.36, 15.8]
radii_kitti[800000] = [0.08, 0.16, 0.31, 0.28, 0.24, 14.96]'''
radii_kitti[100000] = [0.04]
radii_kitti[200000] = [0.07]
radii_kitti[400000] = [0.03]
radii_kitti[800000] = [0.08]

#UniformDist
radii_uniform_dist = {}
radii_uniform_dist[400000] = [0.05, 0.2]

#Holds dataset sizes we are experimenting with. Can be increased to millions?
sizes = [100000, 200000, 400000, 800000]



os.system("make sample06-rtow-mixedGeometries")

#KITTI
'''for size in sizes:
	for k in knn_datasize[size]:
		file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/Kitti/Points_"+str(size)+"/"+str(k)+"knn"
		os.makedirs(file_path, exist_ok = True)
		output_path = file_path + "/Outputs/"
		os.makedirs(output_path, exist_ok = True)
		
		for radius in radii_kitti[size]:
			filename = file_path + "/start_radius_"+str(radius)
			for i in range(0,1):
				cmd = "./sample06-rtow-mixedGeometries /home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/TrueKNN/kitti.csv "+str(size)+" "+str(radius)+" "+str(k)+" "+filename+" >> "+output_path+"/start_radius_"+str(radius)+"_output"
				os.system(cmd)


#Average
for size in sizes:
	print("\nDataset Size = "+str(size))
	for k in knn_datasize[size]:
		file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/Kitti/Points_"+str(size)+"/"+str(k)+"knn"
		print("\n k = "+str(k))
		
		for radius in radii_kitti[size]:
			filename = file_path + "/start_radius_"+str(radius)
			with open(filename,'r') as f:
				data = [float(line.rstrip()) for line in f.readlines()]
				f.close()
			mean = float(sum(data))/len(data) if len(data) > 0 else float('nan')
			print("Radius = "+str(radius)+" Avg Time = "+str(mean))
				
				

#3DIono

for size in sizes:
	for k in knn_datasize[size]:
		file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/3DIono/Points_"+str(size)+"/"+str(k)+"knn"
		os.makedirs(file_path, exist_ok = True)
		output_path = file_path + "/Outputs/"
		os.makedirs(output_path, exist_ok = True)
		
		for radius in radii_3diono[size]:
			filename = file_path + "/start_radius_"+str(radius)
			for i in range(0,10):
				cmd = "./sample06-rtow-mixedGeometries /home/min/a/nagara16/fast-cuda-gpu-dbscan/CUDA_DCLUST_datasets/3D_iono.txt "+str(size)+" "+str(radius)+" "+str(k)+" "+filename+" >> "+output_path+"/start_radius_"+str(radius)+"_output"
				os.system(cmd)


#Average
for size in sizes:
	print("\nDataset Size = "+str(size))
	for k in knn_datasize[size]:
		file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/3DIono/Points_"+str(size)+"/"+str(k)+"knn"
		print("\n k = "+str(k))
		
		for radius in radii_3diono[size]:
			filename = file_path + "/start_radius_"+str(radius)
			with open(filename,'r') as f:
				data = [float(line.rstrip()) for line in f.readlines()]
				f.close()
			mean = float(sum(data))/len(data) if len(data) > 0 else float('nan')
			print("Radius = "+str(radius)+" Avg Time = "+str(mean))


#Porto
for size in sizes:
	for k in knn_datasize[size]:
		file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/Porto/Points_"+str(size)+"/"+str(k)+"knn"
		os.makedirs(file_path, exist_ok = True)
		output_path = file_path + "/Outputs/"
		os.makedirs(output_path, exist_ok = True)
		
		for radius in radii_porto[size]:
			filename = file_path + "/start_radius_"+str(radius)
			for i in range(0,1):
				cmd = "./sample06-rtow-mixedGeometries /home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/porto.csv "+str(size)+" "+str(radius)+" "+str(k)+" "+filename+" >> "+output_path+"/start_radius_"+str(radius)+"_output"
				os.system(cmd)


#Average
for size in sizes:
	file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/Porto/Points_"+str(size)+"/"+str(k)+"knn"
	print("\n Dataset Size = "+str(size))
	
	for radius in radii_porto[size]:
		filename = file_path + "/start_radius_"+str(radius)
		with open(filename,'r') as f:
			data = [float(line.rstrip()) for line in f.readlines()]
			f.close()
		mean = float(sum(data))/len(data) if len(data) > 0 else float('nan')
		print("Radius = "+str(radius)+" Avg Time = "+str(mean))

'''
#UniformDist
for size in sizes:
	for k in knn_datasize[size]:
		file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/UniformDist/Points_"+str(size)+"/"+str(k)+"knn"
		os.makedirs(file_path, exist_ok = True)
		output_path = file_path + "/Outputs/"
		os.makedirs(output_path, exist_ok = True)
		
		for radius in radii_porto[size]:
			filename = file_path + "/start_radius_"+str(radius)
			for i in range(0,1):
				cmd = "./sample06-rtow-mixedGeometries /home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/3D_uniform_dist.csv "+str(size)+" "+str(radius)+" "+str(k)+" "+filename+" >> "+output_path+"/start_radius_"+str(radius)+"_output"
				os.system(cmd)


#Average
for size in sizes:
	file_path = "/home/min/a/nagara16/Downloads/owl/build/MyResults/TrueKNN/UniformDist/Points_"+str(size)+"/"+str(k)+"knn"
	print("\n Dataset Size = "+str(size))
	
	for radius in radii_porto[size]:
		filename = file_path + "/start_radius_"+str(radius)
		with open(filename,'r') as f:
			data = [float(line.rstrip()) for line in f.readlines()]
			f.close()
		mean = float(sum(data))/len(data) if len(data) > 0 else float('nan')
		print("Radius = "+str(radius)+" Avg Time = "+str(mean))

#make sample06-rtow-mixedGeometries && ./sample06-rtow-mixedGeometries /home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/3droad_full.csv 434874 0.08 660 a.txt
#Convert string name to var name: radii_porto = locals()["radii_porto_"+str(size)]
