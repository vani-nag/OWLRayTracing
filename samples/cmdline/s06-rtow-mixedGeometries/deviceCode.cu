// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// #include "deviceCode.h"
#include "GeomTypes.h"
#include <optix_device.h>
#include <bits/stdc++.h>
// here
// #include</home/min/a/nagara16/owl/owl/include/owl/common/parallel/parallel_for.h>

using namespace owl;

#define NUM_SAMPLES_PER_PIXEL 16

// here
__constant__ MyGlobals optixLaunchParams;
#define FLOAT_MIN 1.175494351e-38
#define FLOAT_MAX 3.402823466e+38
__device__ int intersections = 0;
// ==================================================================
// bounding box programs - since these don't actually use the material
// they're all the same irrespective of geometry type, so use a
// template ...
// ==================================================================
template <typename SphereGeomType>
inline __device__ void boundsProg(const void *geomData,
								  box3f &primBounds,
								  const int primID)
{
	const SphereGeomType &self = *(const SphereGeomType *)geomData;
	const Sphere sphere = self.prims[primID];
	primBounds = box3f()
					 .extend(sphere.center - self.rad)
					 .extend(sphere.center + self.rad);
}

OPTIX_BOUNDS_PROGRAM(Spheres)
(const void *geomData,
 box3f &primBounds,
 const int primID)
{
	boundsProg<SpheresGeom>(geomData, primBounds, primID);
}

// ==================================================================
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================
OPTIX_INTERSECT_PROGRAM(Spheres) ()
{
	// ID of the sphere the ray intersected
	const int primID = optixGetPrimitiveIndex();

	// ID of the ray
	int xID = optixGetLaunchIndex().x;

	// The number of neighbors as sepcified by the user
	int k = optixLaunchParams.k;

	// Count number of intersections
	optixLaunchParams.frameBuffer[xID * k].intersections += 1;

	// Check if we have already processed this sphere in a previous iteration. If check == 1, we've seen it before
	int check = 0;
	for (int i = 0; i < k; i++) {
		if (optixLaunchParams.frameBuffer[xID * k + i].ind == primID) {
			// if(xID == 6)
			// printf("INTERSECT: Already intersected %d\n",primID);
			check = 1;
			break;
		}
	}

	if (check == 0) {

		// Access the array of spheres
		const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();

		// Access intersected sphere using its ID
		Sphere self = selfs.prims[primID];

		/*
		The frameBuffer is arranged as: [0,1,..,k-2,k-1, k,k+1..2k-1, 2k..3k-1], where neighbors of sphere 0 are stored at [0..k-1], sphere 1 at [k..2k-1] etc..
		The last element for each subgroup contains the neighbor at the maximum distance. 
		Example: for sphere 0, sphere k-1 is the most distant neighbor; for sphere 1, sphere 2k-1 etc..
		*/
		float maxDist = optixLaunchParams.frameBuffer[xID * k + k - 1].dist;

		// Avoid self-intersections where xID == primID as it will always have distance = 0
		if (xID != primID) {
			// Get coordinates of center of sphere (point in the original dataset)
			const vec3f org = optixGetWorldRayOrigin();
			float x, y, z;

			// Get x2-x1, y2-y1, z2-z1
			// Calculate Euclidean distance between ray and intersected object
			x = self.center.x - org.x;
			y = self.center.y - org.y;
			z = self.center.z - org.z;
			float distance = std::sqrt((x * x) + (y * y) + (z * z));

			// Check if distance to currently intersceted sphere is less than the max we have seen so far
			if (distance < maxDist) {
				// Smaller than max distance, so it will be a new neighbor
				if (optixLaunchParams.frameBuffer[xID * k].numNeighbors > 0)
					optixLaunchParams.frameBuffer[xID * k].numNeighbors -= 1;

				int q = 0, w = k - 1;
				for (; q < k; q++) {
					// Handle frameBuffer values from previous iterations: if the distance and index of neighbor is same => have already seen this before
					// Need to figure out where to insert the current point
					if (distance < optixLaunchParams.frameBuffer[xID * k + q].dist) 
						break;
				}
				// Update the neighbor list
				for (; w > q; w--) {
					optixLaunchParams.frameBuffer[xID * k + w].dist = optixLaunchParams.frameBuffer[xID * k + w - 1].dist;
					optixLaunchParams.frameBuffer[xID * k + w].ind = optixLaunchParams.frameBuffer[xID * k + w - 1].ind;
				}
				optixLaunchParams.frameBuffer[xID * k + w].dist = distance;
				optixLaunchParams.frameBuffer[xID * k + w].ind = primID;
			}
		}
	}
}

OPTIX_RAYGEN_PROGRAM(rayGen) ()
{

	const RayGenData &self = owl::getProgramData<RayGenData>();
	vec3f color = 0.f;
	int xID = optixGetLaunchIndex().x;
	int knn = optixLaunchParams.k;

	// Only launch rays if the point hasn't found k nearest neighbors yet. We use xID*k so that point 0 has neighbors from 0 to k-1 etc..
	if (optixLaunchParams.frameBuffer[xID * knn].numNeighbors > 0) {
		owl::Ray ray(optixLaunchParams.spheres[xID].center, vec3f(0, 0, 1), 0, 1.e-16f);
		owl::traceRay(self.world, ray, color);
	}
}