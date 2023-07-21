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

#pragma once

#include <owl/owl.h>
#include "barnesHutTree.h"
#include <vector>
#include <chrono>

constexpr int NUM_POINTS = 1000000;

using namespace owl;
using namespace std;

  struct NodePersistenceInfo {
    Node bhNode;
    uint8_t dontTraverse;

    NodePersistenceInfo() {
      dontTraverse = 0;
    }

    NodePersistenceInfo(Node node, uint8_t value) {
      bhNode = node;
      dontTraverse = value;
    }
  };

  // ==================================================================
  /* the raw geometric shape of a sphere, without material - this is
     what goes into intersection and bounds programs */
  // ==================================================================
  struct Sphere {
    vec3f center;
    float mass;
  };

  struct SpheresGeom {
    Sphere *prims;
    float rad;
  };

  // ==================================================================
  /* and finally, input for raygen and miss programs */
  // ==================================================================
  struct RayGenData
  {
    OptixTraversableHandle *worlds;
    Point *points;
  };

  struct PerRayData
  {
    int intersections;
  };

	struct MyGlobals 
	{	
		int yIDx;
    int xIDxOffset;
    int parallelLaunch;
    u_int *outputIntersectionData;
    long int *nodesPerLevel;
	};

  struct ProfileStatistics {
    chrono::microseconds intersectionsTime;
    chrono::microseconds sceneBuildTime;
    chrono::microseconds totalProgramTime;
    chrono::microseconds treeBuildTime;
    chrono::microseconds forceCalculationTime;
    chrono::microseconds cpuForceCalculationTime;
    chrono::microseconds intersectionsSetupTime;

    ProfileStatistics() : intersectionsTime(0), sceneBuildTime(0), totalProgramTime(0), treeBuildTime(0), forceCalculationTime(0), cpuForceCalculationTime(0), 
    intersectionsSetupTime(0) {}

  };

