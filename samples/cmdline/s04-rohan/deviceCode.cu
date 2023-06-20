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

// __device__ inline char atomicOr(char *const address, char const val) {
//   char const long_address_modulo = reinterpret_cast< size_t >( address ) & 0x3;
//   u_int *const baseAddress = reinterpret_cast< u_int * >( reinterpret_cast< size_t >( address ) - long_address_modulo ); 
//   u_int constexpr byteSelection[] = {0x3214, 0x3240, 0x3410, 0x4210};
//   u_int const byteSelector = byteSelection[long_address_modulo];
//   u_int long_old, long_assumed, long_val, replacement;

//   long_old = *baseAddress;
//   do {
//     long_assumed = long_old;
//     long_val = __byte_perm(long_old, 0, long_address_modulo) | ;
//     u_int const comparsion = __byte_perm(longOldValue, longCompare, byteSelector);

//     longAssumed = longOldValue;
    
//     longOldValue = ::atomicCAS(baseAddress, comparsion, replacement);
//     oldValue = (longOldValue >> (longAddressModulo * 8)) & 0xFF;

//   } while (compare == oldValue and longAssumed != longOldValue);

//   return oldValue;

// }

__device__ inline char getBitAtPositionInBitmap(char *bitmap, unsigned long position) {
  unsigned long bytePosition = position / 8;
  unsigned long bitPosition = position % 8;

  //int byte = bitmap[bytePosition];
  char bit = (bitmap[bytePosition] >> bitPosition) & 1;
  return bit;
}


__device__ inline void setBitAtPositionInBitmap(char *bitmap, unsigned long position, char value) {
  unsigned long bytePosition = position / 8;
  unsigned long bitPosition = position % 8;

  char *bytePtr = &bitmap[bytePosition];
  int* intPtr = reinterpret_cast<int*>(bytePtr);

  if(((*bytePtr >> bitPosition) & 1) != value) { 
    int bitToSet = value << bitPosition;
    bytePtr[bytePosition] |= bitToSet;
    //atomicOr(intPtr, bitToSet);
  }

   bytePtr = reinterpret_cast<char*>(intPtr);

  // char byte = bitmap[bytePosition];
  // char bit = (byte >> bitPosition) & 1;
  // if(bit != value) {
  //   bitmap[bytePosition] ^= (1 << bitPosition);
  // }
  //printf("The value at position %lu is %d\n", position, getBitAtPositionInBitmap(bitmap, position));
}

OPTIX_INTERSECT_PROGRAM(Spheres)()
{ 
	const int primID = optixGetPrimitiveIndex();
	int xID = optixGetLaunchIndex().x;
  int level = optixGetLaunchIndex().y + 1;
  if(optixLaunchParams.parallelLaunch == 0) {
    level = optixLaunchParams.yIDx;
  }
  const SpheresGeom &selfs = owl::getProgramData<SpheresGeom>();
  Sphere self = selfs.prims[primID];
  float radius = selfs.rad;
  PerRayData &prd = owl::getPRD<PerRayData>();
  
  
  //Inside circle?
  const vec3f org = optixGetWorldRayOrigin();
  float x,y,z;
  
  //Get closest hit triangle's associated circle
  x = self.center.x - org.x;
  y = self.center.y - org.y;
  z = self.center.z - org.z;

  long *nodesPerLevel = optixLaunchParams.nodesPerLevel;
  //int *offsetPerLevel = optixLaunchParams.offsetPerLevel;
  //printf("index: %d\n", ((xID * nodesPerLevel[level]) + primID));
  if(std::sqrt((x*x) + (y*y) + (z*z)) <= radius)
	{
    setBitAtPositionInBitmap(optixLaunchParams.outputIntersectionData, ((xID * nodesPerLevel[level]) + primID), 1);
    unsigned int test1 = 0x12345678;
    unsigned int test2 = 0x9ABCDEF0;
    unsigned int selector = 0x0003;
    uint32_t result = __byte_perm(test1, 0, selector);
    printf("Result: %x\n", result);
    //printf("Byte position: %lu and value %d,  Bit position: %lu and value %d\n", bytePosition, byte, bitPosition, bit);
    // unsigned long bytePosition = ((xID * nodesPerLevel[level]) + primID) / 8;
    // unsigned long bitPosition = ((xID * nodesPerLevel[level]) + primID) % 8;

    // char byte = optixLaunchParams.outputIntersectionData[bytePosition];
    // char bit = (byte >> bitPosition) & 1;
    // if(bit != 1) {
    //  printf("Byte position: %lu and value %d,  Bit position: %lu and value %d\n", bytePosition, byte, bitPosition, bit);
    //   //optixLaunchParams.outputIntersectionData[bytePosition] == (1 << bitPosition);
    //   optixLaunchParams.outputIntersectionData[bytePosition] = 1;
    // }
    //optixLaunchParams.outputIntersectionData[((xID * nodesPerLevel[level]) + primID)] = 1;
    //printf("Ray %d in level %d intersected circle with center x = %f, y = %f, z = %f , mass = %f\n", xID, level, self.center.x, self.center.y, self.center.z, self.mass);
    //printf("Ray %d in level %d with %lu nodes intersected primID: %d \n", xID, level, nodesPerLevel[level] , primID);
  }
  //printf("At idx: %d, Value of bitmap: %d\n", ((xID * nodesPerLevel[level]) + primID), getBitAtPositionInBitmap(optixLaunchParams.outputIntersectionData, ((xID * nodesPerLevel[level]) + primID)));
    
}

// ==================================================================
// miss and raygen
// ==================================================================

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	const RayGenData &self = owl::getProgramData<RayGenData>();
	int xID = optixGetLaunchIndex().x;
  int yID = optixGetLaunchIndex().y + 1;
	owl::Ray ray(vec3f(self.points[xID].x,self.points[xID].y,0), vec3f(0,0,1), 0, 1.e-16f);
  PerRayData prd;

  if(optixLaunchParams.parallelLaunch == 0) {
    yID = optixLaunchParams.yIDx;
  }

  owl::traceRay(self.worlds[yID], ray, prd);
  
}

