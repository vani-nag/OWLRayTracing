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

#include "GeomTypes.h"
#include <optix_device.h>
#include</home/min/a/nagara16/Downloads/owl/owl/include/owl/common/parallel/parallel_for.h>
#define ind 5000
#define radius 2
using namespace owl;
#define NUM_SAMPLES_PER_PIXEL 16
__device__ int a[ind],b[ind];
__device__ int lock = 0;
__device__ int cluster_num = 0;

__device__ DisjointSet ds;
__constant__ MyGlobals optixLaunchParams;


////////////////////////////////////////////////////////////////DISJOINT SET/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



__device__ int find(int x)
{
	while (optixLaunchParams.frameBuffer[x].parent != x) 
	{
		optixLaunchParams.frameBuffer[x].parent = optixLaunchParams.frameBuffer[optixLaunchParams.frameBuffer[x].parent].parent;
		x = optixLaunchParams.frameBuffer[x].parent;
	}

	//printf("Leaving find()\n");
	return 	x;
}

/*__device__ void Union(int x, int y)
{
	// Find current sets of x and y
	int xset = find(x);
	int yset = find(y);

	// If they are already in same set
	if (xset == yset)
		return;

	// Put smaller ranked item under
	// bigger ranked item if ranks are
	// different
	if (optixLaunchParams.frameBuffer[xset].rank < optixLaunchParams.frameBuffer[yset].rank) 
	{
		optixLaunchParams.frameBuffer[xset].parent = yset;
	}
	else if (optixLaunchParams.frameBuffer[xset].rank > optixLaunchParams.frameBuffer[yset].rank) 
	{
		optixLaunchParams.frameBuffer[yset].parent = xset;
	}

	// If ranks are same, then increment
	// rank.
	else {
		optixLaunchParams.frameBuffer[yset].parent = xset;
		optixLaunchParams.frameBuffer[xset].rank = optixLaunchParams.frameBuffer[xset].rank + 1;
	}
	//printf("After union: parent[%d] = %d, parent[%d] = %d\n",x, optixLaunchParams.frameBuffer[xset].parent, y, optixLaunchParams.frameBuffer[yset].parent);
}
*/


////////////////////////////////////////////////////////////////CODE BEGINS//////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
	//printf("Radius = %f\n",self.rad);
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
// SAXPY
// ==================================================================
__global__ void saxpy(int n, float a, int *x, int *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}


// ==================================================================
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================
OPTIX_INTERSECT_PROGRAM(Spheres)()
{ 
	const int primID = optixGetPrimitiveIndex();
	int xID = optixGetLaunchIndex().x;
  const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();
  Sphere self = selfs.prims[primID];
  
  //Inside circle?
  const vec3f org = optixGetWorldRayOrigin();
  float x,y,z;
  
  //Get closest hit triangle's associated circle
  x = self.center.x - org.x;
  y = self.center.y - org.y;
	z = self.center.z - org.z;
	
	//printf("callNum = %d\n",optixLaunchParams.callNum);
	
	if(std::sqrt((x*x) + (y*y) + (z*z)) <= selfs.rad)
	{
		if(optixLaunchParams.callNum == 1)		
		{
			//printf("[%d] hits %d at %f\n",optixGetLaunchIndex().x, primID, std::sqrt((x*x) + (y*y) + (z*z)));
			optixLaunchParams.frameBuffer[xID].isCore = optixLaunchParams.frameBuffer[xID].isCore + 1;
		}
		
		
///////////////////////////////////////////////////////////CALL-2////////////////////////////////////////////////////////////////
	
	
		if(optixLaunchParams.callNum == 2)
		{
			if(optixLaunchParams.frameBuffer[xID].isCore == 1)
			{
				int xset = find(xID);
				if(optixLaunchParams.frameBuffer[primID].isCore == 1)
				{
						int yset = find(primID), xrank = optixLaunchParams.frameBuffer[xset].rank, yrank = optixLaunchParams.frameBuffer[yset].rank;
						//printf("Before: CORE %d hits %d:\n  parent[%d] = %d\n\n", optixGetLaunchIndex().x, primID, primID, find(primID));
						if (xset != yset)
						{
							if (xrank < yrank) 
								optixLaunchParams.frameBuffer[xset].parent = yset;
							else if (xrank > yrank) 
								optixLaunchParams.frameBuffer[yset].parent = xset;
							else 
							{
								optixLaunchParams.frameBuffer[yset].parent = xset;
								optixLaunchParams.frameBuffer[xset].rank = xrank + 1;
							}
						}
					//printf("After: CORE %d hits %d:\n  parent[%d] = %d \n\n", optixGetLaunchIndex().x, primID, primID, find(primID));
				}
					
					
				//////////////////////////////////////Critical section//////////////////////////////////////////////////////////////////
				else
				{
					//__syncthreads();
					//lock();
					//Union(x,y);
					//unlock();
					//printf("CAS for %d:\n Before: parent[%d] = %d\n\n", optixGetLaunchIndex().x, primID, find(primID));
					//atomicCAS(&(optixLaunchParams.frameBuffer[primID].parent), find(primID), find(optixGetLaunchIndex().x)); 
					
					atomicCAS(&(optixLaunchParams.frameBuffer[primID].parent), primID, xset);
					//printf("CAS for %d: \n After: parent[%d] = %d\n\n", optixGetLaunchIndex().x, primID, find(primID));
				}
			}				
		}
	}  
}


/*! transform a _point_ from object to world space */
inline __device__ vec3f pointToWorld(const vec3f &P)
{
  return (vec3f)optixTransformPointFromObjectToWorldSpace(P);
}
inline __device__
vec3f missColor(const Ray &ray)
{
  const vec2i pixelID = owl::getLaunchIndex(); 
  const vec3f rayDir = normalize(ray.direction);
  const float t = 0.5f*(rayDir.y + 1.0f);
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_MISS_PROGRAM(miss)()
{

  PerRayData &prd = owl::getPRD<PerRayData>();

}





// ==================================================================
// miss and raygen
// ==================================================================

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	const RayGenData &self = owl::getProgramData<RayGenData>();
	vec3f color = 0.f;  
	int xID = optixGetLaunchIndex().x;
	owl::Ray ray(self.spheres[xID].center, vec3f(0,0,1), 0, 2);
	//printf("Starting ray %d\n", optixGetLaunchIndex().x);
	
	
	////////////////////////////////////////////////////////CALL-1/////////////////////////////////////////////////////////////
	
	if(optixLaunchParams.callNum == 1)
	{		
		//Launch ray
		owl::traceRay(self.world, ray, color);
		
		//Update core point
		if (optixLaunchParams.frameBuffer[xID].isCore >= optixLaunchParams.minPts)
		{
			//printf("%d is a core point = %d\n",optixGetLaunchIndex().x,optixLaunchParams.frameBuffer[xID].isCore);
			optixLaunchParams.frameBuffer[xID].isCore = 1;
		}
		else
			optixLaunchParams.frameBuffer[xID].isCore = 0;
	}
		
		
	///////////////////////////////////////////////////////////CALL-2////////////////////////////////////////////////////////////////
	
	if(optixLaunchParams.callNum == 2)
	{	
		//printf("CallNum2 for %d\n", optixGetLaunchIndex().x);
			//printf("Parent of %d = %d\n",i,optixLaunchParams.frameBuffer[i].parent);
		//printf("Checking isCore of %d = %d\n",optixGetLaunchIndex().x, optixLaunchParams.frameBuffer[optixGetLaunchIndex().x].isCore);
		//printf("Launching %d\n", optixGetLaunchIndex().x);
		owl::traceRay(self.world, ray, color);
			
		//printf("After TraceRay: \n");
		/*for(i = 0; i < optixLaunchParams.spheresCount; i++)
		{
			printf("Find(%d) = %d\n",i,find(optixLaunchParams.frameBuffer[i].parent));
			optixLaunchParams.frameBuffer[i].parent = find(optixLaunchParams.frameBuffer[i].parent);
		}*/
		
		//optixLaunchParams.frameBuffer[optixGetLaunchIndex().x].parent = find(optixLaunchParams.frameBuffer[optixGetLaunchIndex().x].parent);
		 		
	}
	//printf("Find(13) = %d\n",find(13));
	//printf("Ending ray %d\n", optixGetLaunchIndex().x);
	//printf("find(6) = %d, find(25) = %d, find(19) = %d\n", find(6),find(25),find(19));
}

