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
#define ind 200000
#include<math.h>
__device__ int a[ind];

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();


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
    
  
  //printf("Origin (%f, %f, %f)\n",ray.origin.x,ray.origin.y,ray.origin.z);
  vec3f color;
  owl::traceRay(self.world,
                ray,
                color);
    
  	for(int i = 0; i < ind; i++)
  		self.fbPtr[i] = a[i];
}


/////////////////////////////////////////ANY HIT////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
OPTIX_ANY_HIT_PROGRAM(tmesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();
  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  const int   primID = optixGetPrimitiveIndex();

  //Inside circle?
  const vec3f org   = optixGetWorldRayOrigin();
  float x,y;

  //Get closest hit triangle's associated circle
 /* x = self.circle[primID].x - org.x;
  y = self.circle[primID].y - org.y;*/

	a[primID] = 1;
  /*if(std::sqrt((x*x) + (y*y)) <= self.radius[0].x)
  	a[primID] = 1;*/

	optixIgnoreIntersection();

  //printf("Hit %d\n\t circle = %f,%f,%f\n\t\t sqrt((x*x) + (y*y)) = %f\n",primID,self.circle[primID].x,self.circle[primID].y,self.circle[primID].z, std::sqrt((x*x) + (y*y)));
  /*if( std::sqrt((x*x) + (y*y)) <= self.radius[primID].x)
	  printf("point %f, %f, %f in circle = %f,%f\n\n",org.x, org.y, org.z, self.circle[primID].x, self.circle[primID].y);*/
  //printf("ANY HIT: %d\n",primID);
  //printf("radius %f\n",self.radius[primID].x);

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  //int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  //prd = (pattern&1) ? self.color1 : self.color0;
}

