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

//#include</home/min/a/nagara16/Downloads/owl/owl/include/owl/common/parallel/parallel_for.h>

using namespace owl;
__constant__ MyGlobals optixLaunchParams;

////////////////////////////////////////////////////////////////CODE BEGINS//////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline char getBitAtPositionInBitmap(char *bitmap, unsigned long position) {
  unsigned long bytePosition = position / 8;
  unsigned long bitPosition = position % 8;

  //int byte = bitmap[bytePosition];
  char bit = (bitmap[bytePosition] >> bitPosition) & 1;
  return bit;
}

__device__ inline u_int deviceSetBitAtPositionInBitmap(u_int *bitmap, unsigned long position, u_int value) {
  unsigned long bytePosition = position / 32;
  unsigned long bitPosition = position % 32;

  u_int *bytePtr = &bitmap[bytePosition];
  u_int bitToSet = (value << bitPosition);

  return atomicOr(bytePtr, bitToSet);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  const int primID = optixGetPrimitiveIndex();
  //printf("primID: %d\n", primID);
  PerRayData &prd = owl::getPRD<PerRayData>();

  // compute force between point and bhNode
  Point point = optixLaunchParams.devicePoints[prd.pointID];
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[primID];

  optixLaunchParams.computedForces[prd.pointID] += (((point.mass * bhNode.mass)) / prd.r_2) * GRAVITATIONAL_CONSTANT;
  // if(prd.pointID == 0) {
  //   printf("%sIntersected yay!%s\n",
  //          OWL_TERMINAL_GREEN,
  //          OWL_TERMINAL_DEFAULT);
  //   printf("Current rtComputedForce: %f\n", optixLaunchParams.computedForces[prd.pointID]);
  // }
}

OPTIX_MISS_PROGRAM(miss)()
{
  const MissProgData &self = owl::getProgramData<MissProgData>();
  // printf("%sMissed it!%s\n",
  //          OWL_TERMINAL_RED,
  //          OWL_TERMINAL_DEFAULT);
  PerRayData &prd = owl::getPRD<PerRayData>();
  int insertIndex = prd.insertIndex;

  if(insertIndex == 0)
    deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[prd.primID];
  else {
    deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[prd.rays[insertIndex-1].primID];
  }
  if(bhNode.numChildren == 0) {
    optixLaunchParams.computedForces[prd.pointID] += (((optixLaunchParams.devicePoints[prd.pointID].mass * bhNode.mass)) / prd.r_2) * GRAVITATIONAL_CONSTANT;
    // if(prd.pointID == 0) {
    // printf("%sHit leaf in miss yay!%s\n",
    //        OWL_TERMINAL_GREEN,
    //        OWL_TERMINAL_DEFAULT);
    // }
  } else {
    CustomRay rayObject;
    rayObject.primID = bhNode.primIds[prd.nextChildIndex[insertIndex]];
    rayObject.pointID = prd.pointID;
    rayObject.orgin = bhNode.children[prd.nextChildIndex[insertIndex]];
    prd.rays[insertIndex] = rayObject;
    prd.nextChildIndex[insertIndex] += 1;
    prd.insertIndex += 1;
    //if(prd.pointID == 0) printf("PrimID: %d\n", rayObject.primID);
  }
  //atomicAdd(optixLaunchParams.raysToLaunch, bhNode.numChildren);

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

  PerRayData prd;
  prd.r_2 = (dx * dx) + (dy * dy);
  prd.pointID = currentRay.pointID;
  prd.primID = currentRay.primID;
  prd.insertIndex = 0;
  float rayLength = sqrtf(prd.r_2) * 0.5f;
  // if(prd.pointID == 0) printf("dx: %f | dy: %f\n", dx, dy);
  //if(prd.pointID == 0) printf("Index: %d | PrimID: %d | rayLength: %f\n", 0, prd.primID, rayLength);

  // Launch rays
  int index = 0;
  owl::Ray ray(currentRay.orgin, vec3f(1,0,0), 0, rayLength);
  while(index <= prd.insertIndex) {
    if(rayLength != 0.0f) {
      owl::traceRay(self.world, ray, prd);
    }

    currentRay = prd.rays[index];
    bhNode = optixLaunchParams.deviceBhNodes[currentRay.primID];

    dx = point.x - bhNode.centerOfMassX;
    dy = point.y - bhNode.centerOfMassY;
    prd.r_2 = (dx * dx) + (dy * dy);
    prd.primID = currentRay.primID;
    rayLength = sqrtf(prd.r_2) * 0.5f;

    ray.origin = currentRay.orgin;
    ray.tmax = rayLength;
    // if(prd.pointID == 0) {
    //   //printf("Index: %d | PrimID: %d | rayLength: %f\n", index, prd.rays[index].primID, rayLength);
    //   printf("insertIndex: %d\n", prd.insertIndex);
    // }
    index++;

  }
}

