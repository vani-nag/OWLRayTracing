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

using namespace owl;
#define NUM_SAMPLES_PER_PIXEL 16
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
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================
OPTIX_INTERSECT_PROGRAM(Spheres)()
{ 
	const int primID = optixGetPrimitiveIndex();
	int xID = optixGetLaunchIndex().x;
  const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();
  Sphere self = selfs.prims[primID];
  int minPts = optixLaunchParams.minPts;
  
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
			optixLaunchParams.frameBuffer[xID].neighCount = optixLaunchParams.frameBuffer[xID].neighCount + 1;
		}
		
		
///////////////////////////////////////////////////////////CALL-2////////////////////////////////////////////////////////////////
	
	
		if(optixLaunchParams.callNum == 2 && xID > primID)
		{
			if(optixLaunchParams.frameBuffer[xID].neighCount >= minPts)
			{
				int xParent = find(xID);
				if(optixLaunchParams.frameBuffer[primID].neighCount >= minPts)
				{
					int primIDParent = find(primID);	
					bool repeat;
					do
					{
						repeat = false;
						if (xParent != primIDParent)
						{
							int ret;
							if (xParent < primIDParent)
							{
								if ((ret = atomicCAS(&(optixLaunchParams.frameBuffer[primIDParent].parent), primIDParent,
																				xParent)) != primIDParent)
								{
									primIDParent = ret;
									repeat = true;
								}
							}
							else
							{
								if ((ret = atomicCAS(&(optixLaunchParams.frameBuffer[xParent].parent), xParent,
																				primIDParent)) != xParent)
								{
									xParent = ret;
									repeat = true;
								}
							}
						}
					} while (repeat);
								
				}
					
					
				//////////////////////////////////////Critical section//////////////////////////////////////////////////////////////////
				
				else	
					 optixLaunchParams.frameBuffer[primID].parent = xParent;
			}		
					
				////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////			
		}
	}  
}






// ==================================================================
// miss and raygen
// ==================================================================

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	const RayGenData &self = owl::getProgramData<RayGenData>();
	vec3f color = 0.f;  
	int xID = optixGetLaunchIndex().x;
	owl::Ray ray(self.spheres[xID].center, vec3f(0,0,1), 0, 1.e-16f);
	//printf("Starting ray %d\n", optixGetLaunchIndex().x);
	
	
	////////////////////////////////////////////////////////CALL-1/////////////////////////////////////////////////////////////
	
	if(optixLaunchParams.callNum == 1)
	{		
		//printf("call 1\n");
		//Launch ray
		owl::traceRay(self.world, ray, color);
		/*if(xID == 15790)
			printf("Neighs = %d\n",optixLaunchParams.frameBuffer[xID].neighCount);
		if(optixLaunchParams.frameBuffer[xID].neighCount > 100)
			//atomicAdd(&optixLaunchParams.frameBuffer[xID].counter,optixLaunchParams.frameBuffer[xID].neighCount-100);*/
			//printf("Neighs[%d] = %d\n",xID,optixLaunchParams.frameBuffer[xID].neighCount);
		//Update core point
		//done in intersect implicitly
	}
	///////////////////////////////////////////////////////////CALL-2////////////////////////////////////////////////////////////////
	
	if(optixLaunchParams.callNum == 2)
	{	
		//printf("CallNum2 for %d\n", optixGetLaunchIndex().x);
			//printf("Parent of %d = %d\n",i,optixLaunchParams.frameBuffer[i].parent);
		//printf("Checking neighCount of %d = %d\n",optixGetLaunchIndex().x, optixLaunchParams.frameBuffer[optixGetLaunchIndex().x].neighCount);
		//printf("Launching %d\n", optixGetLaunchIndex().x);
		owl::traceRay(self.world, ray, color);
			
		//printf("After TraceRay: parent[%d] = %d \n",xID,find(xID));
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

