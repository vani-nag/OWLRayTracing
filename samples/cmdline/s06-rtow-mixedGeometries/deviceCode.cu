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
// here
//#include</home/min/a/nagara16/owl/owl/include/owl/common/parallel/parallel_for.h>

using namespace owl;

#define NUM_SAMPLES_PER_PIXEL 16

//here
__constant__ MyGlobals optixLaunchParams;
#define FLOAT_MIN 1.175494351e-38
#define FLOAT_MAX 3.402823466e+38
__device__ int intersections = 0;
// ==================================================================
// bounding box programs - since these don't actually use the material
// they're all the same irrespective of geometry type, so use a
// template ...
// ==================================================================
template<typename SphereGeomType>
inline __device__ void boundsProg(const void *geomData,
                                  box3f &primBounds,
                                  const int primID)
{   
  const SphereGeomType &self = *(const SphereGeomType*)geomData;
	//printf("BOUNDS: Radius = %f\n",self.rad);
  const Sphere sphere = self.prims[primID];
  primBounds = box3f()
    .extend(sphere.center - self.rad)
    .extend(sphere.center + self.rad);	

}


OPTIX_BOUNDS_PROGRAM(Spheres)(const void  *geomData,
  box3f       &primBounds,
  const int    primID)
{ boundsProg<SpheresGeom>(geomData,primBounds,primID); }





// ==================================================================
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================
OPTIX_INTERSECT_PROGRAM(Spheres)()
{ 
	//ID of the sphere the ray intersected
	const int primID = optixGetPrimitiveIndex();
	
	//ID of the ray
	int xID = optixGetLaunchIndex().x;
	
	//The number of neighbors -- as sepcified by the user
	int k = optixLaunchParams.k;
	
	//Count number of intersections
	intersections += 1;
	
	//Access the array of spheres
	const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();
	//printf("INTERSECT: radius = %f\n",selfs.rad);
	
	//Access intersected sphere using its ID
	Sphere self = selfs.prims[primID];
	
  	//int xID = selfs.prims[optixGetLaunchIndex().x].index;
  	//printf("INTERSECT: xID = %d and primID = %d\n",xID,primID);
  	//printf("xID = %d and optixGetLaunchIndex().x = %d\n",xID, optixGetLaunchIndex().x);
	  //int xID = optixGetLaunchIndex().x;
	  
	/*
	The frameBuffer is arranged as: [0..k-1, k..2k-1, 2k..3k-1], where neighbors of sphere 0 are stored at [0..k-1], sphere 1 at [k..2k-1] etc..
	The last element for each subgroup contains the neighbor at the maximum distance. Ex: for sphere 0, sphere k-1 is the most distant neighbor; for sphere 1, sphere 2k-1 etc.. 
	*/
	float maxDist = optixLaunchParams.frameBuffer[xID*k+k-1].dist;

	//Avoid self-intersections where xID == primID as it will always have distance = 0
	if(xID != primID)
	{
		
		
		//Get coordinates of center of sphere (point in the original dataset)
		const vec3f org = optixGetWorldRayOrigin();
		float x,y,z;
		
		//Get x2-x1, y2-y1, z2-z1
		//Calculate Euclidean distance between ray and intersected object
		x = self.center.x - org.x;
		y = self.center.y - org.y;
		z = self.center.z - org.z;
		float distance = std::sqrt((x*x) + (y*y) + (z*z));
		
		//printf("Before filter\nFound neighbor between index = %d: (%f,%f,%f) and primIndex = %d: (%f,%f,%f) dist = %f \n\n", xID,org.x,org.y,org.z,primID,self.center.x,self.center.y,self.center.z,distance);
		//printf("distance between (%f,%f,%f) and (%f,%f,%f) = %f\n",org.x,org.y,org.z,self.center.x,self.center.y,self.center.z,distance);
		//Filter results with additional check	
		//printf("INTERSECT: maxDist for xID = %d: (%f,%f,%f) = %f\n",xID,org.x,org.y,org.z,maxDist);
		
		/*
		Additional level of testing to ensure that the distance between the 2 sphere centers (points in the original dataset) is within specified limit 
		-- can we remove for true knn?
		*/
		if(distance <= optixLaunchParams.distRadius)//selfs.rad)//
		{
		//printf("INTERSECT: distance between xID = %d: (%f,%f,%f) and primID = %d: (%f,%f,%f) = %f\n",xID,org.x,org.y,org.z,primID,self.center.x,self.center.y,self.center.z,distance);
		//printf("\n\nAfter 1st filter:\nFound neighbor between index = %d: (%f,%f,%f) and primIndex = %d: (%f,%f,%f) \n\t dist = %f maxDist = %f \n", 
			//xID,org.x,org.y,org.z,primID,self.center.x,self.center.y,self.center.z,distance,maxDist);
	
			//Check if distance to currently intersceted sphere is less than the max we have seen so far
			if(distance < maxDist)
			{		
				//Smaller than max distance, so it will be a new neighbor
				if(optixLaunchParams.frameBuffer[xID*k].numNeighbors > 0)
					optixLaunchParams.frameBuffer[xID*k].numNeighbors -= 1;	
				//if(xID == 3019)
  				//printf("INTERSECT: Numneighbors for %d = %d\n",xID, optixLaunchParams.frameBuffer[xID*optixLaunchParams.k].numNeighbors);	
				//printf("Found neighbor between index = %d: (%f,%f,%f) and primIndex = %d: (%f,%f,%f) dist = %f \n", xID,org.x,org.y,org.z,primID,self.center.x,self.center.y,self.center.z,distance);		
				int q=0, w=k-1;
				for(; q<k; q++){
				
					//Need to figure out where to insert the current point
					if(distance < optixLaunchParams.frameBuffer[xID*k+q].dist)
					{
						if(xID == 0)
							printf("Intersect distance = %f | frameBuffer distance = %f | q = %d\n", distance, optixLaunchParams.frameBuffer[xID*k+q].dist, q);
						break;
					}
				}
				for(; w>q; w--){
					optixLaunchParams.frameBuffer[xID*k+w].dist = optixLaunchParams.frameBuffer[xID*k+w-1].dist;
					optixLaunchParams.frameBuffer[xID*k+w].ind = optixLaunchParams.frameBuffer[xID*k+w-1].ind;
				  }
				  optixLaunchParams.frameBuffer[xID*k+w].dist = distance;
				  optixLaunchParams.frameBuffer[xID*k+w].ind = primID;
				  if(xID == 0)
				  	printf("frameBuffer[%d].dist = %f\nframeBuffer[%d].ind = %d\n", xID*k+w, optixLaunchParams.frameBuffer[xID*k+w].dist, xID*k+w,optixLaunchParams.frameBuffer[xID*k+w].ind);
			}
		}	
	}
  	if(xID ==0)
  	{
  		for(int i=0;i<k;i++)
  			printf("optixLaunchParams.frameBuffer[%d].dist = %f\n", i, optixLaunchParams.frameBuffer[i].dist);
  	}
  
}	



OPTIX_RAYGEN_PROGRAM(rayGen)()
{

  const RayGenData &self = owl::getProgramData<RayGenData>();
  vec3f color = 0.f;  
  int xID = optixGetLaunchIndex().x;
  int knn = optixLaunchParams.k;
  //optixLaunchParams.frameBuffer[id*optixLaunchParams.k].numNeighbors = 0;
  
  //Only launch rays if the point hasn't found k nearest neighbors yet. We use xID*k so that point 0 has neighbors from 0 to k-1 etc..
  if(optixLaunchParams.frameBuffer[xID*knn].numNeighbors > 0)
  {
  	//printf("RAYGEN: Launching ray for id = %d xID = %d Point[xID] = (%f,%f,%f) numNeighbors = %d\n",id,xID,optixLaunchParams.spheres[xID].center.x,optixLaunchParams.spheres[xID].center.y,optixLaunchParams.spheres[xID].center.z,optixLaunchParams.frameBuffer[id*optixLaunchParams.k].numNeighbors);
		optixLaunchParams.frameBuffer[xID*optixLaunchParams.k].numNeighbors = knn;
		
		owl::Ray ray(optixLaunchParams.spheres[xID].center, vec3f(0,0,1), 0, 1.e-16f);
 	 	owl::traceRay(self.world, ray, color);
  }
  //if(xID == 3019)
  	//printf("Numneighbors for xID = %d and id = %d: %d\n",xID, id, optixLaunchParams.frameBuffer[3019*optixLaunchParams.k].numNeighbors);

  
  //If ray still hasn't found k nearest neighbors, write 0 to frameBuffer[id*optixLaunchParams.k].foundKNN
  
  
	
	//printf("RAYGEN: Point (%f,%f,%f) | numNeighs = %d\n",optixLaunchParams.spheres[xID].center.x, optixLaunchParams.spheres[xID].center.y, optixLaunchParams.spheres[xID].center.z,optixLaunchParams.frameBuffer[id*optixLaunchParams.k].numNeighbors);
	//printf("Intersections for %d: %d\n", xID, intersections);
	
  
}
