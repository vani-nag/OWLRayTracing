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
//#include</home/min/a/nagara16/Downloads/owl/owl/include/owl/common/parallel/parallel_for.h>

using namespace owl;
__constant__ MyGlobals optixLaunchParams;

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
  int level = optixGetLaunchIndex().y;
  if(optixLaunchParams.parallelLaunch == 0) {
    level = optixLaunchParams.yIDx;
  }
  const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();
  Sphere self = selfs.prims[primID];
  float &prd = getPRD<float>();
  
  
  //Inside circle?
  //const vec3f org = optixGetWorldRayOrigin();
  // float x,y,z;
  
  //Get closest hit triangle's associated circle
  // x = self.center.x - org.x;
  // y = self.center.y - org.y;
  // z = self.center.z - org.z;

  if(self.isLeaf == false) 
    printf("Ray %d in level %d intersected circle with center x = %f, y = %f, z = %f , mass = %f, color = %f\n", xID, level, self.center.x, self.center.y, self.center.z, self.mass, prd);
}

// ==================================================================
// miss and raygen
// ==================================================================

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	const RayGenData &self = owl::getProgramData<RayGenData>();
	float level = 1.0f;  
	int xID = optixGetLaunchIndex().x;
  int yID = optixGetLaunchIndex().y;
	owl::Ray ray(vec3f(self.points[xID].x,self.points[xID].y,0), vec3f(0,0,1), 0, 1.e-16f);
  //printf("Level: %d \n", optixLaunchParams.yIDx);
  //printf("Starting ray for level %d with index: %d\n", yID, xID);
  //printf("Starting ray in level %d at circle with center x = %f, y = %f \n", yID, self.points[xID].x, self.points[xID].y);
  if(optixLaunchParams.parallelLaunch == 0) {
    yID = optixLaunchParams.yIDx;
  }
  //printf("Starting ray for level %d with index: %d\n", yID, xID);
  owl::traceRay(self.worlds[yID], ray, level);
}

