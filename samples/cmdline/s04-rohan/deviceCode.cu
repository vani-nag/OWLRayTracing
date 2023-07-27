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
  printf("primID: %d\n", primID);
  printf("%sIntersected yay!%s\n",
           OWL_TERMINAL_GREEN,
           OWL_TERMINAL_DEFAULT);
}

OPTIX_MISS_PROGRAM(miss)()
{
  const MissProgData &self = owl::getProgramData<MissProgData>();
  printf("%sMissed it!%s\n",
           OWL_TERMINAL_RED,
           OWL_TERMINAL_DEFAULT);

}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  vec3f color;
  owl::Ray ray(vec3f(0,0,0), vec3f(1,0,0), 0, 2.5f);

  owl::traceRay(self.world, ray, color);
}

