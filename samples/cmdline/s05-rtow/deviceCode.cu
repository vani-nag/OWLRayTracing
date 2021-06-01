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
#define ind 500000
using namespace owl;
#define NUM_SAMPLES_PER_PIXEL 16
__device__ int a[ind];
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
	  const Sphere sphere = self.prims[primID];
	  primBounds = box3f()
	    .extend(sphere.center - sphere.radius)
	    .extend(sphere.center + sphere.radius);	
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
  const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();
  Sphere self = selfs.prims[primID];

  //Inside circle?
  const vec3f org   = optixGetWorldRayOrigin();
  float x,y,z;
  
  //Get closest hit triangle's associated circle
  x = self.center.x - org.x;
  y = self.center.y - org.y;
	z = self.center.z - org.z;
  
	//a[self.index] = 1;
  if(std::sqrt((x*x) + (y*y) + (z*z)) <= self.radius)
  	a[self.index] = 1;
  	
	  //printf("point %f, %f, %f in circle = %f,%f\n\n",org.x, org.y, org.z, self.center.x, self.center.y);
  //printf("optix's primitiveID: %d\n", self.index); 
    printf("Hit %d\n\t circle = %f,%f,%f\n\t\t sqrt((x*x) + (y*y) + (z*z)) = %f\n",primID,self.center.x,self.center.y,self.center.z, std::sqrt((x*x) + (y*y) + (z*z)));
  
}


/*! transform a _point_ from object to world space */
inline __device__ vec3f pointToWorld(const vec3f &P)
{
  return (vec3f)optixTransformPointFromObjectToWorldSpace(P);
}

// ==================================================================
// plumbing for closest hit
// ==================================================================



OPTIX_CLOSEST_HIT_PROGRAM(Spheres)()
{ 
  const int primID = optixGetPrimitiveIndex();
	    printf("came hit\n");
  const auto &self
    = owl::getProgramData<SpheresGeom>().prims[primID];
  //printf("Closest hit index from Primitive %d\n",self.index);
  PerRayData &prd = owl::getPRD<PerRayData>();

  const vec3f org  = optixGetWorldRayOrigin();
  const vec3f dir  = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  /* iw, jan 11, 2020: for this particular example (where we do not
     yet use instancing) we could also get away with just using the
     sphere.center value directly (since we don't use instancing the
     transform will not have any effect, anyway); but this version is
     cleaner since it would/will also work with instancing */
  const vec3f N     = hit_P-pointToWorld(self.center); 
}




// ==================================================================
// miss and raygen
// ==================================================================

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


OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
	int count = 0;
  
	printf(".cu origin = %f,%f,%f\n",self.origin.x,self.origin.y,self.origin.z);
  vec3f color = 0.f;

    owl::Ray ray(/* origin   : */ self.origin,
                     /* direction: */ vec3f(0,0,1),
                     /* tmin     : */ 0,
                     /* tmax     :  */2);
    
    	owl::traceRay(/*accel to trace against*/self.world,
		        /*the ray to trace*/ray,
		        /*prd*/color);

  	for(int i = 0; i < ind; i++)
		{
  		self.fbPtr[i] = a[i];
			if(a[i] == 1)
			{
				printf("i = %d\n",i);
				count++;
			}
		}
		printf("Count = %d\n",count);
		if(count < 3)//self.minPts)
	  	for(int i = 0; i < ind; i++)
				self.fbPtr[i] = 0;



  /*self.fbPtr[1] = 1;
    self.fbPtr[3] = 1;
      self.fbPtr[7] = 1;
  	printf("a[0] = %d\n",a[0]);*/
}


