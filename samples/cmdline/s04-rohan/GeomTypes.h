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

constexpr int NUM_POINTS = 100000;

using namespace owl;
using namespace std;

  struct deviceBhNode {
    float mass;
    float s;
    float centerOfMassX;
    float centerOfMassY;
    vec3f nextRayLocation;
    int nextPrimId;
    vec3f autoRopeRayLocation;
    int autoRopePrimId;
    uint8_t isLeaf;
  };

  struct IntersectionResult {
    int index;
    int primID;
    float mass;
    float rayLength;

    IntersectionResult() : index(0), primID(0), mass(0), rayLength(0){}
  };

  /* variables for the triangle mesh geometry */
  struct TrianglesGeomData
  {
    /*! array/buffer of vertex indices */
    vec3i *index;
    /*! array/buffer of vertex positions */
    vec3f *vertex;
  };

  struct CustomRay 
  {
    vec3f orgin;
    int primID;
    int pointID;

    // Constructor
    // CustomRay(vec3f org, int prID, int poID) : orgin(org), primID(prID), pointID(poID) {}
    //CustomRay() : orgin(vec3f(0.0f, 0.0f, 0.0f)), primID(0), pointID(0) {}
  };

  /* variables for the ray generation program */
  struct RayGenData
  {
    OptixTraversableHandle world;
    CustomRay *primaryLaunchRays;
  };

  /* variables for the miss program */
  struct MissProgData
  {
    vec3f  color0;
    vec3f  color1;
  };

  struct PerRayData
  {
    float r_2;
    int pointID;
    int primID;
    CustomRay rayToLaunch;
    uint8_t rayEnd;
    //CustomRay rays[50];
    //uint8_t nextChildIndex[50];
    //int insertIndex; 
  };

	struct MyGlobals 
	{	
    long int *nodesPerLevel;
    deviceBhNode *deviceBhNodes;
    Point *devicePoints;
    int numPrims;
    float *computedForces;
    int *raysToLaunch;
    CustomRay *rayObjectsToLaunch;
	};

  struct ProfileStatistics {
    chrono::microseconds intersectionsTime;
    chrono::microseconds sceneBuildTime;
    chrono::microseconds totalProgramTime;
    chrono::microseconds treeBuildTime;
    chrono::microseconds forceCalculationTime;
    chrono::microseconds cpuForceCalculationTime;
    chrono::microseconds intersectionsSetupTime;
    chrono::microseconds treePathsRecrusiveSetupTime;
    chrono::microseconds treePathsIterativeSetupTime;

    ProfileStatistics() : intersectionsTime(0), sceneBuildTime(0), totalProgramTime(0), treeBuildTime(0), forceCalculationTime(0), cpuForceCalculationTime(0), 
    intersectionsSetupTime(0) {}

  };

