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

#define BYTES_PER_BATCH 4000000000.0 // 6gb max

extern "C" char deviceCode_ptx[];

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

// force calculation global variables
vector<vector<Node>> bhNodes;
int totalNumNodes = 0;
vector<float> computedForces(NUM_POINTS, 0.0f);
vector<float> maxForces(NUM_POINTS, 0.0f);
vector<float> cpuComputedForces(NUM_POINTS, 0.0f);

OWLContext context = owlContextCreate(nullptr, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

// const int NUM_VERTICES = 6;
// vec3f vertices[NUM_VERTICES] =
//   {
//     { +1.f, 0.0f,-0.5f },
//     { +1.f,-0.5f,0.5f },
//     { +1.f,+0.5f,+0.5f },
//     { +2.f, 0.0f,-0.5f },
//     { +2.f,-0.5f,0.5f },
//     { +2.f,+0.5f,+0.5f }
//   };

// const int NUM_INDICES = 2;
// vec3i indices[NUM_INDICES] =
//   {
//     { 0,1,2 },
//     { 3,4,5 }
//   };

// // ##################################################################
// // Create world function
// // ##################################################################
// OptixTraversableHandle createSceneGivenGeometries(vector<Sphere> Spheres, float spheresRadius) {

//   // Create Spheres Buffer
//   OWLBuffer SpheresBuffer = owlDeviceBufferCreate(
//       context, OWL_USER_TYPE(Spheres[0]), Spheres.size(), Spheres.data());

//   OWLGeom SpheresGeometry = owlGeomCreate(context, SpheresGeomType);
//   owlGeomSetPrimCount(SpheresGeometry, Spheres.size());
//   owlGeomSetBuffer(SpheresGeometry, "prims", SpheresBuffer);
//   owlGeomSet1f(SpheresGeometry, "rad", spheresRadius);

//   // Setup accel group
//   OWLGeom userGeoms[] = {SpheresGeometry};

//   OWLGroup spheresGroup = owlUserGeomGroupCreate(context, 1, userGeoms);
//   owlGroupBuildAccel(spheresGroup);

//   OWLGroup world = owlInstanceGroupCreate(context, 1, &spheresGroup);
//   owlGroupBuildAccel(world);

//   size_t final_memory_usage = 0;
//   size_t peak_memory_usage = 0;
//   owlGroupGetAccelSize(world, &final_memory_usage, &peak_memory_usage);

//   printf("Final memory usage is %lu, and peak memory usage is %lu\n", final_memory_usage, peak_memory_usage);

//   //LOG_OK("Built world for grid size: " << gridSize << " and sphere radius : " << spheresRadius);

//   // return created scene/world
//   return owlGroupGetTraversable(world, deviceID);
// }

// float computeObjectsAttractionForce(Point point, Node bhNode) {
//   float mass_one = point.mass;
//   float mass_two = bhNode.mass;

//   // distance calculation
//   float dx = point.x - bhNode.centerOfMassX;
//   float dy = point.y - bhNode.centerOfMassY;
//   float r_2 = (dx * dx) + (dy * dy);

//   return (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);
// }

int main(int ac, char **av) {
  auto total_run_time = chrono::steady_clock::now();
  
  // char *newBitmap = createBitmap(8, true);
  // for(int i = 0; i < 8; i++) {
  //   printf("Value at index: %d is %d\n", i, getBitAtPositionInBitmap(newBitmap, i));
  //   setBitAtPositionInBitmap(newBitmap, i, 1);
  //   printf("Value at index: %d is %d\n", i, getBitAtPositionInBitmap(newBitmap, i));
  // }


  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################
  BarnesHutTree* tree = new BarnesHutTree(THRESHOLD, gridSize);
  Node* root = new Node(0.f, 0.f, gridSize);

  vector<vec3f> vertices;
  vector<vec3i> indices;
  vector<Point> points;
  Point p0 = {.x = -.773f, .y = 2.991f, .mass = 12.213f, .idX=0};
  Point p1 = {.x = -3.599f, .y = -2.265, .mass = 17.859f, .idX=1};
  Point p2 = {.x = -4.861f, .y = -1.514f, .mass = 3.244f, .idX=2};
  Point p3 = {.x = -3.662f, .y = 2.338f, .mass = 13.3419f, .idX=3};
  Point p4 = {.x = -2.097f, .y = 2.779f, .mass = 19.808f, .idX=4};
  points.push_back(p0);
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);
  points.push_back(p4);

  // for (int i = 0; i < NUM_POINTS; ++i) {
  //   Point p;
  //   p.x = dis(gen);
  //   p.y = dis(gen);
  //   p.mass = disMass(gen);
  //   p.idX = i;
  //   points.push_back(p);
  //   //printf("Point # %d has x = %f, y = %f, mass = %f\n", i, p.x, p.y, p.mass);
  // }

  for(int i = 1; i < points.size() + 1; i++) {
    vertices.push_back(vec3f{static_cast<float>(i), 0.0f, -0.5f});
    vertices.push_back(vec3f{static_cast<float>(i), -0.5f, 0.5f});
    vertices.push_back(vec3f{static_cast<float>(i), 0.5f, 0.5f});
  }

  for(int i = 0; i < vertices.size(); i+=3) {
    indices.push_back(vec3i{i, i+1, i+2});
  }

  OWLBuffer PointsBuffer = owlDeviceBufferCreate(
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());


  LOG("Bulding Tree with # Of Bodies = " << points.size());
  auto tree_build_time_start = chrono::steady_clock::now();
  for(const auto& point: points) {
    tree->insertNode(root, point);
  };

  auto tree_build_time_end = chrono::steady_clock::now();
  profileStats->treeBuildTime += chrono::duration_cast<chrono::microseconds>(tree_build_time_end - tree_build_time_start);

  tree->printTree(root, 0, "root");

  // Get the device ID
  cudaGetDevice(&deviceID);
  LOG_OK("Device ID: " << deviceID);
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
    {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  // vector<Node> InternalNodes;
  // float prevS = gridSize;
  // int level = 0;

  // // level order traversal of BarnesHutTree
  // // Create an empty queue for level order traversal
  // queue<Node*> q;
  
  // LOG("Bulding OWL Scenes");
  // // Enqueue Root and initialize height
  // q.push(root);
  auto scene_build_time_start = chrono::steady_clock::now();
  // float currentForces = 0;
  // while (q.empty() == false) {
  //     // Print front of queue and remove it from queue
  //     Node* node = q.front();
  //     if((node->s != prevS)) {
  //       if(!InternalSpheres.empty()) {
  //         worlds.push_back(createSceneGivenGeometries(InternalSpheres, (gridSize / THRESHOLD)));
  //         // if(!offsetPerLevel.empty()) {
  //         //   offsetPerLevel.push_back((offsetPerLevel.back() + (NUM_POINTS * nodesPerLevel.back())));
  //         // } else {
  //         //   offsetPerLevel.push_back(0);
  //         // }
  //         nodesPerLevel.push_back(InternalSpheres.size());
  //         if(maxNodesPerLevel < InternalSpheres.size()) {
  //           maxNodesPerLevel = InternalSpheres.size();
  //         }
  //         bhNodes.push_back(InternalNodes);
  //       } else {
  //         LOG_OK("Spheres r empty!");
  //       }
  //       InternalSpheres.clear();
  //       InternalNodes.clear();
  //       prevS = node->s;
  //       gridSize = node->s;
  //       level += 1;
  //     } else {
  //       //LOG_OK("HITS THIS!");
  //     }
  //     if(node->s == gridSize) {
  //       if(node->mass != 0.0f) {
  //         if(node->ne == nullptr) {
  //           node->isLeaf = true;
  //         } else {
  //           node->isLeaf = false;
  //         }
  //         InternalNodes.push_back((*node));
  //         totalNumNodes += 1;
  //         InternalSpheres.push_back(Sphere{vec3f{node->centerOfMassX, node->centerOfMassY, 0}, node->mass});
  //       }
  //     }
  //     q.pop();

  //     /* Enqueue left child */
  //     if (node->nw != NULL)
  //         q.push(node->nw);

  //     /*Enqueue right child */
  //     if (node->ne != NULL)
  //         q.push(node->ne);
      
  //     /* Enqueue left child */
  //     if (node->sw != NULL)
  //         q.push(node->sw);

  //     /*Enqueue right child */
  //     if (node->se != NULL)
  //         q.push(node->se);
  // }

  // // last level gets missed in while loop so this is to account for that
  // if(!InternalSpheres.empty()) {
  //   worlds.push_back(createSceneGivenGeometries(InternalSpheres, (gridSize / THRESHOLD)));
  //   // if(!offsetPerLevel.empty()) {
  //   //   offsetPerLevel.push_back((offsetPerLevel.back() + (NUM_POINTS * nodesPerLevel.back())));
  //   // } else {
  //   //   offsetPerLevel.push_back(0);
  //   // }
  //   nodesPerLevel.push_back(InternalSpheres.size());
  //   if(maxNodesPerLevel < InternalSpheres.size()) {
  //     maxNodesPerLevel = InternalSpheres.size();
  //   }
  //   bhNodes.push_back(InternalNodes);
  // }
  
  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
      //{"internalSpheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, internalSpheres)},
      {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
      {/* sentinel to mark end of list */}};

  // ........... create object  ............................
  OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                                     sizeof(RayGenData), rayGenVars, -1);             

  // ----------- set variables  ----------------------------
  owlRayGenSetGroup(rayGen, "world", world);

  // programs have been built before, but have to rebuild raygen and
  // miss progs

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  auto scene_build_time_end = chrono::steady_clock::now();
  profileStats->sceneBuildTime += chrono::duration_cast<chrono::microseconds>(scene_build_time_end - scene_build_time_start);


  // ##################################################################
  // Start Ray Tracing Parallel launch
  // ##################################################################
  auto start1 = std::chrono::steady_clock::now();
  owlLaunch2D(rayGen, 1, 1, lp);

  auto end1 = std::chrono::steady_clock::now();
  auto elapsed1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cout << "Intersections time for parallel launch: " << elapsed1.count() / 1000000.0
            << " seconds." << std::endl;
  

  // ##################################################################
  // Output Force Computations
  // ##################################################################
  // compute real forces using cpu BH traversal
  auto cpu_forces_start_time = chrono::steady_clock::now();
  tree->computeForces(root, points, cpuComputedForces);
  auto cpu_forces_end_time = chrono::steady_clock::now();
  profileStats->cpuForceCalculationTime += chrono::duration_cast<chrono::microseconds>(cpu_forces_end_time - cpu_forces_start_time);


  // for(int i = 0; i < NUM_POINTS; i++) {
  //   float percent_error = (abs((computedForces[i] - cpuComputedForces[i])) / cpuComputedForces[i]) * 100.0f;
  //   if(percent_error > .5f) {
  //     LOG_OK("++++++++++++++++++++++++");
  //     LOG_OK("POINT #" << i << ", (" << points[i].x << ", " << points[i].y << ") , HAS ERROR OF " << percent_error << "%");
  //     LOG_OK("++++++++++++++++++++++++");
  //     printf("RT force = %f\n", computedForces[i]);
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