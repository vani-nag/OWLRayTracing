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

using namespace owl;
__constant__ MyGlobals optixLaunchParams;

////////////////////////////////////////////////////////////////CODE BEGINS//////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  //const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  //const int primID = optixGetPrimitiveIndex();

  // compute force between point and bhNode
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[optixGetPayload_4()];
  float totalMass = __uint_as_float(optixGetPayload_1());
  float currentMass = (((__uint_as_float(optixGetPayload_2()) * __uint_as_float(optixGetPayload_3()))) / __uint_as_float(optixGetPayload_7())) * .0001f;

  optixSetPayload_1(__float_as_uint((totalMass + currentMass)));
  optixSetPayload_4(bhNode.autoRopePrimId);
  optixSetPayload_5(__float_as_uint(bhNode.autoRopeRayLocation.x));
  optixSetPayload_6(__float_as_uint(bhNode.autoRopeRayLocation.y));
  //if(prd.pointID == 5382) printf("Approximated at node with mass! ->%f\n", bhNode.mass);
}

OPTIX_MISS_PROGRAM(miss)()
{
  //const MissProgData &self = owl::getProgramData<MissProgData>();
  //PerRayData &prd = owl::getPRD<PerRayData>();
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[optixGetPayload_4()];
  float totalMass = __uint_as_float(optixGetPayload_1());
  float currentMass = 0.0f;
  
  if(bhNode.isLeaf == 1 && optixGetPayload_0() == 0) {
    currentMass = (((__uint_as_float(optixGetPayload_2()) * __uint_as_float(optixGetPayload_3()))) / __uint_as_float(optixGetPayload_7())) * .0001f;
    optixSetPayload_1(__float_as_uint((totalMass + currentMass)));
    //prd.mass += (((__uint_as_float(optixGetPayload_2()) * __uint_as_float(optixGetPayload_3()))) / __uint_as_float(optixGetPayload_7())) * GRAVITATIONAL_CONSTANT;
    //if(prd.pointID == 5382) printf("Intersected leaf at node with mass! ->%f\n", bhNode.mass);
  } 
  optixSetPayload_4(bhNode.nextPrimId);
  optixSetPayload_5(__float_as_uint(bhNode.nextRayLocation.x));
  optixSetPayload_6(__float_as_uint(bhNode.nextRayLocation.y));

  //printf("Ray distance %f.\n", prd.r_2);

}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  CustomRay currentRay = self.primaryLaunchRays[pixelID.x];
  Point point = optixLaunchParams.devicePoints[currentRay.pointID];
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[currentRay.primID];

  // Calculate distance between point and bh node
  float dx = fabs(point.x - bhNode.centerOfMassX);
  float dy = fabs(point.y - bhNode.centerOfMassY);
  float dz = fabs(point.z - bhNode.centerOfMassZ);

  int pointID = currentRay.pointID;
  float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
  uint8_t rayEnd = 0;

  unsigned int p0 = 0;                            // raySelf
  unsigned int p1 = __float_as_uint(0.0f);
  unsigned int p2 = __float_as_uint(point.mass);  // point.mass
  unsigned int p3 = __float_as_uint(bhNode.mass); // bhNode.mass
  unsigned int p4 = currentRay.primID;            // primID
  unsigned int p5 = __float_as_uint(currentRay.orgin.x);  // rayOrigin.x
  unsigned int p6 = __float_as_uint(currentRay.orgin.y);  // rayOrigin.y
  unsigned int p7 = __float_as_uint(r_2);                 // r_2
  //rd.hit = 0; 
  //prd.insertIndex = 0;
  float rayLength = sqrtf(r_2) * 0.5f;
  rayLength = sqrtf(rayLength);
  //if(prd.pointID == 0) printf("Num prims %d\n", optixLaunchParams.numPrims);
  //if(prd.pointID == 5382) printf("Index: %d | PrimID: %d | Mass: %f | rayLength: %f\n", 0, prd.primID, bhNode.mass, rayLength);

  int index = 0;
  // Launch rays
  owl::Ray ray(currentRay.orgin, vec3f(1,0,0), 0, rayLength);
  while(rayEnd == 0) {
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
    p3 = __float_as_uint(bhNode.mass);

    dx = point.x - bhNode.centerOfMassX;
    dy = point.y - bhNode.centerOfMassY;
    dz = point.z - bhNode.centerOfMassZ;
    r_2 = (dx * dx) + (dy * dy) + (dz * dz);
    rayLength = sqrtf(r_2) * 0.5f;
    p7 = __float_as_int(r_2);
    rayLength = sqrtf(rayLength);

    //if(index == 10000) optixReorder();

    //ray.origin = vec3f(__uint_as_float(p5), __uint_as_float(p6), 0.0f);
    ray.tmax = rayLength;
    rayEnd = (p4 >= optixLaunchParams.numPrims || p4 == 0) ? 1 : 0;
    //optixReorder(rayEnd, 1);
    p0 = (rayLength == 0.0f) ? 1 : 0;
    //p3 = __float_as_uint(bhNode.mass);
    // if(prd.pointID == 8124) {
    //   IntersectionResult result;
    //   result.index = index;
    //   result.primID = prd.primID;
    //   result.mass = bhNode.mass;
    //   result.rayLength = rayLength;
    //   //result.didIntersect = 1;
    //   prd.result = result;
    //   //optixLaunchParams.intersectionResults[prd.index] = result;
    // }
    // if(prd.pointID == 5382) {
    //   //printf("Index: %d | PrimID: %d | Mass: %f | rayLength: %f | Origin: (%f, %f)\n", index, prd.primID, bhNode.mass, rayLength, ray.origin.x, ray.origin.y);
    // }
    //break;
    //optixReorder(rayEnd, 1);
    index++;
  }
  optixLaunchParams.computedForces[pointID] = __uint_as_float(p1);
}