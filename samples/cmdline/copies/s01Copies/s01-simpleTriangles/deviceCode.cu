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

#include "deviceCode.h"
#include <optix_device.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>
#define low_eps 1.e-16f

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program%s\n",
           OWL_TERMINAL_CYAN,
           OWL_TERMINAL_DEFAULT);
  }

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  vec3f origin = self.origin;
  /*owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  //printf("Origin (%f, %f, %f)\n",ray.origin.x,ray.origin.y,ray.origin.z);
  ray.direction = normalize(self.camera.dir_00+ screen.u * self.camera.dir_du+ screen.v * self.camera.dir_dv);
  //ray.direction = vec3f(0,0,-1);*/

	
    owl::Ray ray(/* origin   : */ origin,
			vec3f(0,0,1),
                     //normalize(self.camera.dir_00+ screen.u * self.camera.dir_du+ screen.v * self.camera.dir_dv),
                     /* tmin     : */ 0,
                     /* tmax     : */ 2);
    
  
  printf("Direction (%f, %f, %f)\n",ray.direction.x,ray.direction.y,ray.direction.z);
  vec3f color;
  owl::traceRay(self.world,
                ray,
                color);
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}


/////////////////////////////////////////ANY HIT////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
OPTIX_ANY_HIT_PROGRAM(tmesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  //printf("ANY HIT: %d\n",primID);
  
  //Inside circle?
  const vec3f org   = optixGetWorldRayOrigin();
  float x,y;

  //Get closest hit triangle's associated circle
  x = self.circle[primID].x - org.x;
  y = self.circle[primID].y - org.y;
  printf("Hit %d\n\t circle = %f,%f,%f\n\t\t (x*x) + (y*y) = %f\n",primID,self.circle[primID].x,self.circle[primID].y,self.circle[primID].z, (x*x) + (y*y));
  if( (x*x) + (y*y) <= 1)
	  printf("point %f, %f, %f in circle = %f,%f\n\n",org.x, org.y, org.z, self.circle[primID].x, self.circle[primID].y);
	optixIgnoreIntersection();
  

}
/////////////////////////////////////////CLOSEST HIT////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  //printf("CLOSEST hit %d\n\t circle = %f,%f\n\n",primID,self.circle[primID].x,self.circle[primID].y);
  
  //Inside circle?
  const vec3f org   = optixGetWorldRayOrigin();
  float x,y,z;

  //Get closest hit triangle's associated circle
  x = self.circle[primID].x - org.x;
  y = self.circle[primID].y - org.y;
  z = self.circle[primID].z - org.z;
  printf("CLOSEST hit %d\n\t circle = %f,%f,%f\n\t\t (x*x) + (y*y) + (z*z) = %f\n",primID,self.circle[primID].x,self.circle[primID].y,self.circle[primID].z, (x*x) + (y*y) + (z*z));
  if( ((x*x) + (y*y) + (z*z)) <= 1)
	  printf("point %f, %f, %f in circle = %f,%f\n\n",org.x, org.y, org.z, self.circle[primID].x, self.circle[primID].y);

  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f N      = normalize(cross(B-A,C-A));

  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  printf("tmax = %f\n",hit_t);

  //prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*self.color;

/////////////////////////////////////////////////NEW RAY////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /*const RayGenData &rayself = owl::getProgramData<RayGenData>();
  owl::Ray ray(hit_P, dir, 0, 2);
  vec3f color;
	printf("Hit point = %f,%f,%f\n",hit_P.x, hit_P.y, hit_P.z);
  owl::traceRay(rayself.world,
                ray,
                color);*/


}


OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  //int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  //prd = (pattern&1) ? self.color1 : self.color0;
}

