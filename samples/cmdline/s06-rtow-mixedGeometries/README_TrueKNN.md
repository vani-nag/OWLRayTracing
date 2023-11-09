# TrueKNN: RT-Accelerated k-Nearest Neighbor Search
To run TrueKNN, use the following command:
```
make sample06-rtow-mixedGeometries
./sample06-rtow-mixedGeometries [filename] [number of points] [dimension] [start radius] [k] [output file]
```
where
- filename: .csv file
- number of inputs (n): number of points to query from dataset. Always selects first *n* points
- dimension: input data can be 2D or 3D. For 1D, you will need to modify code to set both the y and z dimensions to 0 
- start radius: Output of


[random_sample.py] (owl/samples/cmdline/s01-simpleTriangles/testing/random_sample.py)
This is the first file to run to get the start radius.
Change the file path and number of points to read in Line 4.
We need to use `Min distance = ` displayed in the output as start radius

[true_knn_run.py] (https://github.com/vani-nag/OWLRayTracing/blob/master/build/true_knn_run.py)
This is the script to collect data from TrueKNN runs
Line 57 has entries to be filled in for UniformDist dataset.
I filled in Line 59 with the output from random_sample.py and the maximum distance between any 2 points in the first 400K points of the dataset
For the maximum distance part, you can run [knn_euclidean_dist.py] (https://github.com/vani-nag/OWLRayTracing/blob/master/samples/cmdline/s01-simpleTriangles/testing/TrueKNN/knn_euclidean_dist.py). Will just need to change dataset path and nrows
Make sure to save the results! I'll need the 99th percentile value also
The next entry for the dictionary in true_knn_run.py will be radii_uniform_dist[]=  [output of random_sample, max dist from knn_euclidean_dist.py]
Mainly just make sure that the file paths are right. Everything else should be straightforward
Just FYI:
owl/samples/cmdline/s06-rtow-mixedGeometries/
hostCode.cpp, deviceCode --  this is where I have the TrueKNN code