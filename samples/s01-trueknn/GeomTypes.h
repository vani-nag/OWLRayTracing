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

#include "Materials.h"

namespace owl {


	typedef struct maximumDistanceLogger
	{
		int ind;
		float dist;
		int numNeighbors;
		long long int intersections;
	}Neigh;
	


  // ==================================================================
  /* the raw geometric shape of a sphere, without material - this is
     what goes into intersection and bounds programs */
  // ==================================================================
  struct Sphere {
    vec3f center;
  };

 

  // ==================================================================
  /* the three actual "Geoms" that each consist of multiple prims of
     same type (this is what optix6 would have called the "geometry
     instance" */
  // ==================================================================

  struct SpheresGeom {
    Sphere *prims;
    float rad;
  };
 

  // ==================================================================
  /* and finally, input for raygen and miss programs */
  // ==================================================================
  struct RayGenData
  {
    uint32_t *fbPtr;
    vec2i  fbSize;
    OptixTraversableHandle world;
    int sbtOffset;
    //Sphere *spheres;
    
    struct {
      vec3f origin;
      vec3f lower_left_corner;
      vec3f horizontal;
      vec3f vertical;
    } camera;
  };

  struct MissProgData
  {
    /* nothing in this example */
  };

	struct MyGlobals 
	{	
		Neigh *frameBuffer;
		int k;
		Sphere *spheres;
		float distRadius;
	};

}