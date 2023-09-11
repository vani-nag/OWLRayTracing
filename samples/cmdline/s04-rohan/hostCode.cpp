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

// The Ray Tracing in One Weekend scene.
// See https://github.com/raytracing/InOneWeekend/releases/ for this free book.

// public owl API
#include <owl/owl.h>
#include <owl/DeviceMemory.h>

// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <vector>
#include<iostream>
#include <cmath>
#include<fstream>
#include <queue>
#include<string>
#include<sstream>
#include <random>
#include<ctime>
#include<chrono>
#include<algorithm>
#include<set>
#include <random>
// barnesHutStuff
#include "barnesHutTree.h"
#include "bitmap.h"

#define LOG(message)                                            \
  cout << OWL_TERMINAL_BLUE;                               \
  cout << "#owl.sample(main): " << message << endl;    \
  cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  cout << "#owl.sample(main): " << message << endl;    \
  cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

#define RAYS_ARRAY_SIZE 4000000000.0 // 6gb max

// random init
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dis(-5.0f, 5.0f);  // Range for X and Y coordinates
uniform_real_distribution<float> disMass(0.1f, 20.0f);  // Range mass 

// global variables
int deviceID;
float gridSize = GRID_SIZE;
vector<long int> nodesPerLevel;
long int maxNodesPerLevel = 0;
ProfileStatistics *profileStats = new ProfileStatistics();
u_int *deviceOutputIntersectionData;

// force calculation global variables
vector<CustomRay> primaryLaunchRays(NUM_POINTS);
vector<deviceBhNode> deviceBhNodes;
deviceBhNode *deviceBhNodesPointer;
Point *devicePoints;
int totalNumNodes = 0;
vector<float> computedForces(NUM_POINTS, 0.0f);
vector<float> maxForces(NUM_POINTS, 0.0f);
vector<float> cpuComputedForces(NUM_POINTS, 0.0f);

OWLContext context = owlContextCreate(nullptr, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

int main(int ac, char **av) {
  auto total_run_time = chrono::steady_clock::now();

  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################
  BarnesHutTree* tree = new BarnesHutTree(THRESHOLD, gridSize);
  Node* root = new Node(0.f, 0.f, gridSize);

  vector<vec3f> vertices;
  vector<vec3i> indices;
  vector<Point> points;
  // Point p0 = {.x = -.773f, .y = 2.991f, .mass = 12.213f, .idX=0};
  // Point p1 = {.x = -3.599f, .y = -2.265, .mass = 17.859f, .idX=1};
  // Point p2 = {.x = -4.861f, .y = -1.514f, .mass = 3.244f, .idX=2};
  // Point p3 = {.x = -3.662f, .y = 2.338f, .mass = 13.3419f, .idX=3};
  // Point p4 = {.x = -2.097f, .y = 2.779f, .mass = 19.808f, .idX=4};
  // points.push_back(p0);
  // points.push_back(p1);
  // points.push_back(p2);
  // points.push_back(p3);
  // points.push_back(p4);
  // primaryLaunchRays[0].pointID = 0;
  // primaryLaunchRays[0].primID = 0;
  // primaryLaunchRays[0].orgin = vec3f(0.0f, 0.0f, 0.0f);
  // primaryLaunchRays[1].pointID = 1;
  // primaryLaunchRays[1].primID = 0;
  // primaryLaunchRays[1].orgin = vec3f(0.0f, 0.0f, 0.0f);
  // primaryLaunchRays[2].pointID = 2;
  // primaryLaunchRays[2].primID = 0;
  // primaryLaunchRays[2].orgin = vec3f(0.0f, 0.0f, 0.0f);
  // primaryLaunchRays[3].pointID = 3;
  // primaryLaunchRays[3].primID = 0;
  // primaryLaunchRays[3].orgin = vec3f(0.0f, 0.0f, 0.0f);
  // primaryLaunchRays[4].pointID = 4;
  // primaryLaunchRays[4].primID = 0;
  // primaryLaunchRays[4].orgin = vec3f(0.0f, 0.0f, 0.0f);

  for (int i = 0; i < NUM_POINTS; ++i) {
    Point p;
    p.x = dis(gen);
    p.y = dis(gen);
    p.mass = disMass(gen);
    p.idX = i;
    points.push_back(p);
    //printf("Point # %d has x = %f, y = %f, mass = %f\n", i, p.x, p.y, p.mass);
    primaryLaunchRays[i].pointID = i;
    primaryLaunchRays[i].primID = 0;
    primaryLaunchRays[i].orgin = vec3f(0.0f, 0.0f, 0.0f);
  }

  OWLBuffer PointsBuffer = owlDeviceBufferCreate(
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());
  
  OWLBuffer primaryLaunchRaysBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(primaryLaunchRays[0]),
                            primaryLaunchRays.size(),primaryLaunchRays.data());


  LOG("Bulding Tree with # Of Bodies = " << points.size());
  auto tree_build_time_start = chrono::steady_clock::now();
  for(const auto& point: points) {
    tree->insertNode(root, point);
  };

  auto tree_build_time_end = chrono::steady_clock::now();
  profileStats->treeBuildTime += chrono::duration_cast<chrono::microseconds>(tree_build_time_end - tree_build_time_start);

  //tree->printTree(root, 0, "root");

  // Get the device ID
  cudaGetDevice(&deviceID);
  LOG_OK("Device ID: " << deviceID);

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  float prevS = gridSize;
  float nextGridSize = gridSize / 2.0;
  int level = 0;
  long int currentNodesPerLevel = 0;
  int currentIndex = 0;
  float triangleXLocation = 0;
  float rayOriginXLocation = 0;
  int primIDIndex = 1;

  // level order traversal of BarnesHutTree
  // Create an empty queue for level order traversal
  queue<Node*> q;
  
  LOG("Bulding OWL Scenes");
  // Enqueue Root and initialize height
  q.push(root);
  auto scene_build_time_start = chrono::steady_clock::now();
  while (q.empty() == false) {
      // Print front of queue and remove it from queue
      Node* node = q.front();
      if((node->s != prevS)) {
        if(currentNodesPerLevel > maxNodesPerLevel) maxNodesPerLevel = currentNodesPerLevel;
        prevS = node->s;
        gridSize = node->s;
        nextGridSize = gridSize / 2.0;
        level += 1;
        nodesPerLevel.push_back(currentNodesPerLevel);
        currentNodesPerLevel = 0;
        triangleXLocation = 0.0f;
        rayOriginXLocation = 0.0f;
      } 

      if(node->s == gridSize) {
        if(node->mass != 0.0f) {
          if(node->ne == nullptr) {
            node->isLeaf = true;
          } else {
            node->isLeaf = false;
          }
          // add triangle to scene corresponding to barnes hut node
          triangleXLocation += gridSize;
          vertices.push_back(vec3f{static_cast<float>(triangleXLocation), 0.0f + level, -0.5f});
          vertices.push_back(vec3f{static_cast<float>(triangleXLocation), -0.5f + level, 0.5f});
          vertices.push_back(vec3f{static_cast<float>(triangleXLocation), 0.5f + level, 0.5f});
          indices.push_back(vec3i{currentIndex, currentIndex+1, currentIndex+2});
          currentIndex += 3;

          // add device bhNode to vector array
          deviceBhNode currentBhNode;
          int indexToInsertAt = 0;
          currentBhNode.mass = node->mass;
          currentBhNode.s = node->s;
          currentBhNode.centerOfMassX = node->centerOfMassX;
          currentBhNode.centerOfMassY = node->centerOfMassY;
          if(node->nw && node->nw->mass != 0.0f) {
            currentBhNode.children[indexToInsertAt] = vec3f(static_cast<float>(rayOriginXLocation), 0.0f + level + 1, 0.0f);
            currentBhNode.primIds[indexToInsertAt] = primIDIndex;
            rayOriginXLocation += nextGridSize;
            indexToInsertAt++;
            primIDIndex++;
          }
          if(node->ne && node->ne->mass != 0.0f) {
            currentBhNode.children[indexToInsertAt] = vec3f(static_cast<float>(rayOriginXLocation), 0.0f + level + 1, 0.0f);
            currentBhNode.primIds[indexToInsertAt] = primIDIndex;
            rayOriginXLocation += nextGridSize;
            indexToInsertAt++;
            primIDIndex++;
          }
          if(node->sw && node->sw->mass != 0.0f) {
            currentBhNode.children[indexToInsertAt] = vec3f(static_cast<float>(rayOriginXLocation), 0.0f + level + 1, 0.0f);
            currentBhNode.primIds[indexToInsertAt] = primIDIndex;
            rayOriginXLocation += nextGridSize;
            indexToInsertAt++;
            primIDIndex++;
          }
          if(node->se && node->se->mass != 0.0f) {
            currentBhNode.children[indexToInsertAt] = vec3f(static_cast<float>(rayOriginXLocation), 0.0f + level + 1, 0.0f);
            currentBhNode.primIds[indexToInsertAt] = primIDIndex;
            rayOriginXLocation += nextGridSize;
            indexToInsertAt++;
            primIDIndex++;
          }
          currentBhNode.numChildren = indexToInsertAt;
          deviceBhNodes.push_back(currentBhNode);

          // update counters
          totalNumNodes++; currentNodesPerLevel++;
        }
      }
      q.pop();

      /* Enqueue left child */
      if (node->nw != NULL)
          q.push(node->nw);

      /*Enqueue right child */
      if (node->ne != NULL)
          q.push(node->ne);
      
      /* Enqueue left child */
      if (node->sw != NULL)
          q.push(node->sw);

      /*Enqueue right child */
      if (node->se != NULL)
          q.push(node->se);
  }
  
  // last level of BH tree check gets missed so this is to account for it
  nodesPerLevel.push_back(currentNodesPerLevel);
  if(currentNodesPerLevel > maxNodesPerLevel) maxNodesPerLevel = currentNodesPerLevel;

  printf("Max nodes per level = %lu\n", maxNodesPerLevel);

  for(int i = 0; i < nodesPerLevel.size(); i++) {
    printf("Level %d -- %lu nodes\n", i, nodesPerLevel[i]);
  }

  OWLBuffer deviceBhNodesBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(deviceBhNodes[0]),
                            deviceBhNodes.size(),deviceBhNodes.data());

  // create Triangles geomtery and create acceleration structure
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { /* sentinel to mark end of list */ }
  };

  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,-1);

  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");
  
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context, OWL_USER_TYPE(vertices[0]), vertices.size(), vertices.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context, OWL_USER_TYPE(indices[0]), indices.size(), indices.data());
  
  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);
  
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,vertices.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,indices.size(),sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  OWLGroup trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  OWLGroup world
    = owlInstanceGroupCreate(context,1,&trianglesGroup);
  owlGroupBuildAccel(world);

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);

  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});

  OWLVarDecl myGlobalsVars[] = {
    {"nodesPerLevel", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, nodesPerLevel)},
    {"deviceBhNodes", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, deviceBhNodes)},
    {"devicePoints", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, devicePoints)},
    {"level", OWL_INT, OWL_OFFSETOF(MyGlobals, level)},
    {"computedForces", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, computedForces)},
    {"raysToLaunch", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, raysToLaunch)},
    {"rayObjectsToLaunch", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, rayObjectsToLaunch)},
    {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);

  OWLBuffer NodesPerLevelBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(nodesPerLevel[0]), nodesPerLevel.size(), nodesPerLevel.data());
  owlParamsSetBuffer(lp, "nodesPerLevel", NodesPerLevelBuffer);

  OWLBuffer DevicePointsBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());
  owlParamsSetBuffer(lp, "devicePoints", DevicePointsBuffer);

  OWLBuffer DeviceBhNodesBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(deviceBhNodes[0]), deviceBhNodes.size(), deviceBhNodes.data());
  owlParamsSetBuffer(lp, "deviceBhNodes", DeviceBhNodesBuffer);

  OWLBuffer ComputedForcesBuffer = owlManagedMemoryBufferCreate( 
     context, OWL_FLOAT, NUM_POINTS, nullptr);
  owlParamsSetBuffer(lp, "computedForces", ComputedForcesBuffer);

  int raysToLaunchInit = 0;
  OWLBuffer RaysToLaunchBuffer = owlManagedMemoryBufferCreate( 
     context, OWL_INT, 1, &raysToLaunchInit);
  owlParamsSetBuffer(lp, "raysToLaunch", RaysToLaunchBuffer);

  //printf("Size of custom ray is %lu\n", sizeof(CustomRay));
  u_int numberOfRaysToLaunch = RAYS_ARRAY_SIZE / sizeof(CustomRay);
  OWLBuffer NextLevelRaysToLaunchBuffer = owlManagedMemoryBufferCreate( 
     context, OWL_USER_TYPE(CustomRay), numberOfRaysToLaunch, nullptr);
  owlParamsSetBuffer(lp, "rayObjectsToLaunch", NextLevelRaysToLaunchBuffer);
  

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
      //{"internalSpheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, internalSpheres)},
      {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
      {"primaryLaunchRays",  OWL_BUFPTR, OWL_OFFSETOF(RayGenData,primaryLaunchRays)},
      {/* sentinel to mark end of list */}};

  // ........... create object  ............................
  OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                                     sizeof(RayGenData), rayGenVars, -1);             

  // ----------- set variables  ----------------------------
  owlRayGenSetGroup(rayGen, "world", world);
  owlRayGenSetBuffer(rayGen, "primaryLaunchRays", primaryLaunchRaysBuffer);

  // programs have been built before, but have to rebuild raygen and
  // miss progs

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  auto scene_build_time_end = chrono::steady_clock::now();
  profileStats->sceneBuildTime += chrono::duration_cast<chrono::microseconds>(scene_build_time_end - scene_build_time_start);

  auto intersections_setup_time_start = chrono::steady_clock::now();

  printf("Size of malloc for bhNodes is %.2f gb.\n", (deviceBhNodes.size() * sizeof(deviceBhNode))/1000000000.0);
  printf("Size of malloc for points is %.2f gb.\n", (points.size() * sizeof(Point))/1000000000.0);
  
  auto intersections_setup_time_end = chrono::steady_clock::now();
  profileStats->intersectionsSetupTime += chrono::duration_cast<chrono::microseconds>(intersections_setup_time_end - intersections_setup_time_start);

  // ##################################################################
  // Start Ray Tracing Parallel launch
  // ##################################################################
  auto start1 = std::chrono::steady_clock::now();
  for(int i = 0; i < nodesPerLevel.size(); i++) {
    owlParamsSet1i(lp, "level", i);
    //owlBufferUpload(RaysToLaunchBuffer, &raysToLaunchInit, 0, 1);
    owlLaunch2D(rayGen, NUM_POINTS, 1, lp);
    const CustomRay *raysToLaunchOutput = (const CustomRay *)owlBufferGetPointer(NextLevelRaysToLaunchBuffer,0);
    //printf("Number of rays to launch for level %d is %d.\n", i, *raysToLaunchOutput);
    break;
  }
  const float *rtComputedForces = (const float *)owlBufferGetPointer(ComputedForcesBuffer,0);
  auto end1 = std::chrono::steady_clock::now();
  profileStats->forceCalculationTime = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

  // ##################################################################
  // Output Force Computations
  // ##################################################################
  // compute real forces using cpu BH traversal
  auto cpu_forces_start_time = chrono::steady_clock::now();
  tree->computeForces(root, points, cpuComputedForces);
  auto cpu_forces_end_time = chrono::steady_clock::now();
  profileStats->cpuForceCalculationTime += chrono::duration_cast<chrono::microseconds>(cpu_forces_end_time - cpu_forces_start_time);

  // for(int i = 0; i < NUM_POINTS; i++) {
  //   float percent_error = (abs((rtComputedForces[i] - cpuComputedForces[i])) / cpuComputedForces[i]) * 100.0f;
  //   if(percent_error > .5f) {
  //     LOG_OK("++++++++++++++++++++++++");
  //     LOG_OK("POINT #" << i << ", (" << points[i].x << ", " << points[i].y << ") , HAS ERROR OF " << percent_error << "%");
  //     LOG_OK("++++++++++++++++++++++++");
  //     printf("RT force = %f\n", rtComputedForces[i]);
  //     printf("CPU force = %f\n", cpuComputedForces[i]);
  //     LOG_OK("++++++++++++++++++++++++");
  //     printf("\n");
  //   }
  // }

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  auto total_run_time_end = chrono::steady_clock::now();
  profileStats->totalProgramTime += chrono::duration_cast<chrono::microseconds>(total_run_time_end - total_run_time);

  // free memory
  // cudaFree(deviceOutputIntersectionData);
  // delete[] hostOutputIntersectionData;

  // Print Statistics
  printf("--------------------------------------------------------------\n");
  std::cout << "Tree build time: " << profileStats->treeBuildTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Tree Paths Recursive build time: " << profileStats->treePathsRecrusiveSetupTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Tree Paths Iterative build time: " << profileStats->treePathsIterativeSetupTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Scene build time: " << profileStats->sceneBuildTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Intersections setup time: " << profileStats->intersectionsSetupTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Intersections time: " << profileStats->intersectionsTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "RT Cores Force Calculations time: " << profileStats->forceCalculationTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "CPU Force Calculations time: " << profileStats->cpuForceCalculationTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Total Program time: " << profileStats->totalProgramTime.count() / 1000000.0 << " seconds." << std::endl;
  printf("--------------------------------------------------------------\n");

  // destory owl stuff
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
}