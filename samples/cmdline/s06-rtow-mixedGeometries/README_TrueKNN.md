# TrueKNN: RT-Accelerated k-Nearest Neighbor Search
TrueKNN computes the *k* nearest neighbors of *every* point in the dataset.
To run TrueKNN, use the following command:
```
cd build
make sample06-rtow-mixedGeometries
./sample06-rtow-mixedGeometries [filename] [number of points] [dimension] [start radius] [k] [output file]
```
where
- **filename:** .csv file with 2D/3D points
- **number of inputs (*n*):** Number of points to query from dataset. Always selects first *n* points
- **dimension:** Input data can be 2D or 3D. For 1D, you will need to modify code to set both the y and z dimensions to 0 
- **start radius:** Start radius for neighbor search. See [Random Sampling For Start Radius](https://github.com/vani-nag/OWLRayTracing/blob/master/samples/cmdline/s06-rtow-mixedGeometries/README_TrueKNN.md#random-sampling-for-start-radius) for more details on how to get this value.
- **k:** Number of neighbors per point 
- **output file:** file to write execution time 

### Random Sampling For Start Radius
Run [random_sample.py](owl/samples/cmdline/s01-simpleTriangles/testing/random_sample.py) to get the start radius:
`python random_sample.py`
Change the file path and number of points to read in Line 4.
We need to use `Min distance = ` displayed in the output as start radius

