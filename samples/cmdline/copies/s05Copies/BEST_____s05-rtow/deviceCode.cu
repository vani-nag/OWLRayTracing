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
#define low_eps 1.e-16f
using namespace owl;

#define NUM_SAMPLES_PER_PIXEL 16

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
	  printf("optix's primitiveID: %d\n", primID);
PerRayData &prd = owl::getPRD<PerRayData>();
	
	prd.index[primID] = primID;
	  const auto &self
	    = owl::getProgramData<SpheresGeom>().prims[primID];
	printf("Intersection Index: %d\n",self.index);
	  /* iw, jan 11, 2020: for this particular example (where we do not
	     yet use instancing) we could also use the World ray; but this
	     version is cleaner since it would/will also work with
	     instancing 
		CHECK: OBJECT OR WORLD FOR ORIGIN?*/
	  const vec3f org  = optixGetObjectRayOrigin();
	  const vec3f dir  = optixGetObjectRayDirection();
	  float hit_t      = optixGetRayTmax();
	  const float tmin = optixGetRayTmin();

	  const vec3f oc = org - self.center;
	  const float a = dot(dir,dir);
	  const float b = dot(oc, dir);
	  const float c = dot(oc, oc) - self.radius * self.radius;
	  const float discriminant = b * b - a * c;
	  
	  if (discriminant < 0.f) return;

	  {
	    float temp = (-b - sqrtf(discriminant)) / a;
	    if (temp < hit_t && temp > tmin) 
	      hit_t = temp;
	  }
	      
	  {
	    float temp = (-b + sqrtf(discriminant)) / a;
	    if (temp < hit_t && temp > tmin) 
	      hit_t = temp;
	  }

	  if (hit_t < optixGetRayTmax()) {
	    printf("report intersection for %d\n",self.index);
	    optixReportIntersection(hit_t, 0);
	  }

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

  const auto &self
    = owl::getProgramData<SpheresGeom>().prims[primID];
  printf("Closest hit index from Primitive %d\n",self.index);
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
 const int primID = optixGetPrimitiveIndex();

  PerRayData &prd = owl::getPRD<PerRayData>();
	printf("primID from PRD should be 3: %d\n",prd.index[3]);
  prd.out.scatterEvent = rayDidntHitAnything;
}




OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  
   const int pixelIdx = pixelID.x+self.fbSize.x*(self.fbSize.y-1-pixelID.y);
  vec3f color = 0.f;

    owl::Ray ray(/* origin   : */ vec3f(0.f,0.f,0.f),
                     /* direction: */ vec3f(low_eps,low_eps,low_eps),
                     /* tmin     : */ 0,
                     /* tmax     : */ low_eps);
    
   
   
//ray.origin = vec3f(-4.f,0.f,0.f);
//ray.direction= vec3f(low_eps,low_eps,low_eps);
    	owl::traceRay(/*accel to trace against*/self.world,
		        /*the ray to trace*/ray,
		        /*prd*/color);
  //}
    //printf("sampl ID: %d\n",sampleID);
  self.fbPtr[pixelIdx]
    = owl::make_rgba(color * (1.f / NUM_SAMPLES_PER_PIXEL));
}


