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

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  const int primID = optixGetPrimitiveIndex();
  //printf("primID: %d\n", primID);
  PerRayData &prd = owl::getPRD<PerRayData>();

  // compute force between point and bhNode
  Point point = optixLaunchParams.devicePoints[prd.pointID];
  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[primID];

  //if(prd.pointID == 0) printf("Current rtComputedForce: %f\n", optixLaunchParams.computedForces[prd.pointID]);
  optixLaunchParams.computedForces[prd.pointID] += (((point.mass * bhNode.mass)) / prd.r_2) * GRAVITATIONAL_CONSTANT;
  // if(prd.pointID == 8124) {
  //     prd.result.didIntersect = 1;
  //     prd.result.isLeaf = 0;
  //     optixLaunchParams.intersectionResults[prd.result.index] = prd.result;
  //   }
  CustomRay rayObject;
  rayObject.primID = bhNode.autoRopePrimId;
  rayObject.orgin = bhNode.autoRopeRayLocation;
  rayObject.pointID = prd.pointID;
  prd.rayToLaunch = rayObject;
  //if(prd.pointID == 5382) printf("Approximated at node with mass! ->%f\n", bhNode.mass);
  // if(prd.pointID == 8124) {
  // printf("%sIntersected yay!%s\n",
  //          OWL_TERMINAL_GREEN,
  //          OWL_TERMINAL_DEFAULT);
  // // printf("Current rtComputedForce: %f\n", optixLaunchParams.computedForces[prd.pointID]);
  // }
}

OPTIX_MISS_PROGRAM(miss)()
{
  const MissProgData &self = owl::getProgramData<MissProgData>();
  // printf("%sMissed it!%s\n",
  //          OWL_TERMINAL_RED,
  //          OWL_TERMINAL_DEFAULT);
  PerRayData &prd = owl::getPRD<PerRayData>();

  deviceBhNode bhNode = optixLaunchParams.deviceBhNodes[prd.primID];;
  
  if(bhNode.isLeaf == 1) {
    optixLaunchParams.computedForces[prd.pointID] += (((optixLaunchParams.devicePoints[prd.pointID].mass * bhNode.mass)) / prd.r_2) * GRAVITATIONAL_CONSTANT;
    // if(prd.pointID == 8124) {
    //   prd.result.didIntersect = 0;
    //   prd.result.isLeaf = 1;
    //   optixLaunchParams.intersectionResults[prd.result.index] = prd.result;
    // }
  //if(prd.pointID == 5382) printf("Intersected leaf at node with mass! ->%f\n", bhNode.mass);
    // if(prd.pointID == 8124) {
    // printf("%sHit leaf in miss yay!%s\n",
    //        OWL_TERMINAL_GREEN,
    //        OWL_TERMINAL_DEFAULT); }
  } else {
    // if(prd.pointID == 8124) {
    //   prd.result.didIntersect = 0;
    //   prd.result.isLeaf = 0;
    //   optixLaunchParams.intersectionResults[prd.result.index] = prd.result;
    //   //printf("insertIndex: %d\n", prd.result.index);
    // }
    //printf("PrimID: %d\n", prd.primID);
  }
  CustomRay rayObject;
  rayObject.primID = bhNode.nextPrimId;
  rayObject.pointID = prd.pointID;
  rayObject.orgin = bhNode.nextRayLocation;
  prd.rayToLaunch = rayObject;
  prd.rayEnd = 0;
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
  prd.rayEnd = 0;
  //prd.insertIndex = 0;
  float rayLength = sqrtf(prd.r_2) * 0.5f;
  //if(prd.pointID == 0) printf("Num prims %d\n", optixLaunchParams.numPrims);
  //if(prd.pointID == 5382) printf("Index: %d | PrimID: %d | Mass: %f | rayLength: %f\n", 0, prd.primID, bhNode.mass, rayLength);

  // Launch rays
  int index = 0;
  prd.index = index;
  prd.rayLength = rayLength;
  owl::Ray ray(currentRay.orgin, vec3f(1,0,0), 0, rayLength);
  while(prd.rayEnd == 0) {
    if(rayLength != 0.0f) {
      owl::traceRay(self.world, ray, prd);
    } else {
      CustomRay rayObject;
      rayObject.primID = bhNode.nextPrimId;
      rayObject.pointID = prd.pointID;
      rayObject.orgin = bhNode.nextRayLocation;
      prd.rayToLaunch = rayObject;
    }

    currentRay = prd.rayToLaunch;
    bhNode = optixLaunchParams.deviceBhNodes[currentRay.primID];

    dx = point.x - bhNode.centerOfMassX;
    dy = point.y - bhNode.centerOfMassY;
    prd.r_2 = (dx * dx) + (dy * dy);
    prd.primID = currentRay.primID;
    rayLength = sqrtf(prd.r_2) * 0.5;

    ray.origin = currentRay.orgin;
    ray.tmax = rayLength;
    if(prd.primID >= optixLaunchParams.numPrims || prd.primID == 0) {
      prd.rayEnd = 1;
    }
    index++;
    prd.index = index;
    prd.rayLength = rayLength;
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
  }
}

