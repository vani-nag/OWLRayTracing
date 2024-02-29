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
#include "bitmap.h"
#include <cmath>
#include "barnesHutTree.h"

using namespace owl;
__constant__ MyGlobals optixLaunchParams;

////////////////////////////////////////////////////////////////CODE BEGINS//////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  // compute force between point and bhNode
  printf("Hit!\n");
  int pointID = optixGetPayload_3();
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[optixGetPayload_4()];
  float currentComputedForce = __uint_as_float(optixGetPayload_1());
  if(bhNode.isLeaf == 1) {
    for(int i = 0; i < bhNode.numParticles; i++) {
      int particleID = bhNode.particles[i];
      if(particleID != pointID) {
        Point point = optixLaunchParams.devicePoints[particleID];
        float dx = point.pos.x - optixLaunchParams.devicePoints[pointID].pos.x;
        float dy = point.pos.y - optixLaunchParams.devicePoints[pointID].pos.y;
        float dz = point.pos.z - optixLaunchParams.devicePoints[pointID].pos.z;
        float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
        float force = ((point.mass * optixLaunchParams.devicePoints[pointID].mass) / r_2) * .0001f;
        currentComputedForce += force;
      }
    }
  } else {
    currentComputedForce += ((__uint_as_float(optixGetPayload_2()) * bhNode.mass) / __uint_as_float(optixGetPayload_7())) * .0001f;
  }

  optixSetPayload_1(__float_as_uint((currentComputedForce)));
  optixSetPayload_4(bhNode.autoRopePrimId);
  optixSetPayload_5(__float_as_uint(bhNode.autoRopeRayLocation_x));
  optixSetPayload_6(__float_as_uint(bhNode.autoRopeRayLocation_y));
}

OPTIX_MISS_PROGRAM(miss)()
{
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[optixGetPayload_4()];
  float totalMass = __uint_as_float(optixGetPayload_1());
  float currentMass = 0.0f;
  
  if(bhNode.isLeaf == 1 && optixGetPayload_0() == 0) {
    currentMass = (((__uint_as_float(optixGetPayload_2()) * bhNode.mass)) / __uint_as_float(optixGetPayload_7())) * .0001f;
    optixSetPayload_1(__float_as_uint((totalMass + currentMass)));
  } 
  optixSetPayload_4(bhNode.nextPrimId);
  optixSetPayload_5(__float_as_uint(bhNode.nextRayLocation_x));
  optixSetPayload_6(__float_as_uint(bhNode.nextRayLocation_y));
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  CustomRay currentRay = self.primaryLaunchRays[pixelID.x];
  Point point = optixLaunchParams.devicePoints[currentRay.pointID];
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[currentRay.primID];

  // Calculate distance between point and bh node
  float dx = fabs(point.pos.x - bhNode.centerOfMassX);
  float dy = fabs(point.pos.y - bhNode.centerOfMassY);
  float dz = fabs(point.pos.z - bhNode.centerOfMassZ);

  int pointID = currentRay.pointID;
  float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
  uint8_t rayEnd = 0;

  unsigned int p0 = 0;                            // raySelf
  unsigned int p1 = __float_as_uint(0.0f);        // computedForce
  unsigned int p2 = __float_as_uint(point.mass);  // point.mass
  unsigned int p3 = currentRay.pointID;           // pointID
  unsigned int p4 = currentRay.primID;            // primID
  unsigned int p5 = __float_as_uint(currentRay.orgin.x);  // rayOrigin.x
  unsigned int p6 = __float_as_uint(currentRay.orgin.y);  // rayOrigin.y
  unsigned int p7 = __float_as_uint(r_2);                 // r_2

  float rayLength = sqrtf(r_2) * THRESHOLD;
  rayLength = sqrtf(rayLength);

  // Launch rays
  owl::Ray ray(currentRay.orgin, vec3f(1,0,0), 0, rayLength);
  while(rayEnd == 0) {
    //
    optixTraverse(self.world,
               (const float3&)ray.origin,
               (const float3&)ray.direction,
               ray.tmin,
               ray.tmax,
               ray.time,
               ray.visibilityMask,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               /*SBToffset    */ray.rayType,
               /*SBTstride    */ray.numRayTypes,
               /*missSBTIndex */ray.rayType,
               p0, p1, p2, p3, p4,
               p5, p6, p7);
    //optixReorder();
    optixInvoke(p0, p1, p2, p3, p4, p5, p6, p7);

    bhNode = optixLaunchParams.deviceBhNodes[p4];

    ray.origin = vec3f(__uint_as_float(p5), __uint_as_float(p6), 0.0f);
    //p3 = __float_as_uint(bhNode.mass);

    dx = point.pos.x - bhNode.centerOfMassX;
    dy = point.pos.y - bhNode.centerOfMassY;
    dz = point.pos.z - bhNode.centerOfMassZ;
    r_2 = (dx * dx) + (dy * dy) + (dz * dz);
    rayLength = sqrtf(r_2) * THRESHOLD;
    p7 = __float_as_int(r_2);
    rayLength = sqrtf(rayLength);

    ray.tmax = rayLength;
    rayEnd = (p4 >= optixLaunchParams.numPrims || p4 == 0) ? 1 : 0;
    p0 = (rayLength == 0.0f) ? 1 : 0;
    //optixReorder(rayEnd, 1);
  }
  optixLaunchParams.computedForces[pointID] = __uint_as_float(p1);
}