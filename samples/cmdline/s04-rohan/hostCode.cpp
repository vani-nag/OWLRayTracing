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

#define LOG(message)                                            \
  cout << OWL_TERMINAL_BLUE;                               \
  cout << "#owl.sample(main): " << message << endl;    \
  cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  cout << "#owl.sample(main): " << message << endl;    \
  cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

// random init
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dis(-5.0f, 5.0f);  // Range for X and Y coordinates
uniform_real_distribution<float> disMass(0.1f, 20.0f);  // Range mass

// global variables
int deviceID;
float gridSize = GRID_SIZE;
uint8_t *deviceOutputIntersectionData;
uint8_t *hostOutputIntersectionData;
uint8_t *zeroOutputIntersectionData;
vector<long int> nodesPerLevel;
long int maxNodesPerLevel = 0;
std::vector<long int> offsetPerLevel;

// force calculation global variables
vector<vector<NodePersistenceInfo>> prevPersistenceInfo;
vector<vector<Node>> bhNodes;
int totalNumNodes = 0;
vector<float> computedForces(NUM_POINTS, 0.0f);
vector<float> cpuComputedForces(NUM_POINTS, 0.0f);

OWLContext context = owlContextCreate(nullptr, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

OWLVarDecl SpheresGeomVars[] = {
    {"prims", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, prims)},
    {"rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom, rad)},
    {/* sentinel to mark end of list */}};

OWLGeomType SpheresGeomType = owlGeomTypeCreate(
    context, OWL_GEOMETRY_USER, sizeof(SpheresGeom), SpheresGeomVars, -1);

// ##################################################################
// Create world function
// ##################################################################
OptixTraversableHandle createSceneGivenGeometries(vector<Sphere> Spheres, float spheresRadius) {

  // Create Spheres Buffer
  OWLBuffer SpheresBuffer = owlDeviceBufferCreate(
      context, OWL_USER_TYPE(Spheres[0]), Spheres.size(), Spheres.data());

  OWLGeom SpheresGeometry = owlGeomCreate(context, SpheresGeomType);
  owlGeomSetPrimCount(SpheresGeometry, Spheres.size());
  owlGeomSetBuffer(SpheresGeometry, "prims", SpheresBuffer);
  owlGeomSet1f(SpheresGeometry, "rad", spheresRadius);

  // Setup accel group
  OWLGeom userGeoms[] = {SpheresGeometry};

  OWLGroup spheresGroup = owlUserGeomGroupCreate(context, 1, userGeoms);
  owlGroupBuildAccel(spheresGroup);

  OWLGroup world = owlInstanceGroupCreate(context, 1, &spheresGroup);
  owlGroupBuildAccel(world);

  //LOG_OK("Built world for grid size: " << gridSize << " and sphere radius : " << spheresRadius);

  // return created scene/world
  return owlGroupGetTraversable(world, deviceID);
}

size_t calculateQuadTreeSize(Node* node) {
    if (node == nullptr) {
        return 0;  // Base case: empty node
    }

    size_t currentNodeSize = sizeof(Node);

    // Recursively calculate the size of each child subtree
    size_t size = currentNodeSize;  // Count the current node itself
    size += calculateQuadTreeSize(node->nw);
    size += calculateQuadTreeSize(node->ne);
    size += calculateQuadTreeSize(node->sw);
    size += calculateQuadTreeSize(node->se);

    return size;
}

float computeObjectsAttractionForce(Point point, Node bhNode) {
  float mass_one = point.mass;
  float mass_two = bhNode.mass;

  // distance calculation
  float dx = point.x - bhNode.centerOfMassX;
  float dy = point.y - bhNode.centerOfMassY;
  float r_2 = (dx * dx) + (dy * dy);

  return (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);
}

void computeForces(uint8_t *intersectionsOutputData, vector<Point> points, int levelIdx) {
  // printf("==========================================\n");
  // for(int i = 1; i < 2; i++) {
  //   printf("++++++++++++++++++++++++++++++++++++++++\n");
  //   printf("Point # %d with x = %f, y = %f, mass = %f\n", i, points[i].x, points[i].y, points[i].mass);
  //   for(int k = 0; k < nodesPerLevel[levelIdx]; k++) {
  //     if(intersectionsOutputData[levelIdx].pointIntersectionInfo[i].didIntersectNodes[k] != 0) {
  //       // Node bhNode = intersectionsOutputData[levelIdx].pointIntersectionInfo[i].bhNodes[k];
  //          Node bhNode = bhNodes[levelIdx][k];
  //       float radius = (GRID_SIZE / pow(2, levelIdx)) / THRESHOLD;
  //       printf("Intersected bhNode with x = %f, y = %f, mass = %f, radius = %f\n", bhNode.centerOfMassX, bhNode.centerOfMassY, bhNode.mass, radius);
  //     }
  //   }
  // }
  // printf("==========================================\n");

  vector<NodePersistenceInfo> currentPersistenceInfo;
  for(int i = 0; i < NUM_POINTS; i++) {
    for(int k = 0; k < nodesPerLevel[levelIdx]; k++) {
      //if(i == 1) printf("Point %d, Node COM = (%f, %f), dontTraverse = %d, didIntersect = %d\n", i, bhNodes[levelIdx][k].centerOfMassX, bhNodes[levelIdx][k].centerOfMassY, prevPersistenceInfo[i][k].dontTraverse, intersectionsOutputData[levelIdx].pointIntersectionInfo[(i * nodesPerLevel[levelIdx]) + k]);
      //Node bhNode = intersectionsOutputData[levelIdx].pointIntersectionInfo[i].bhNodes[k];
      Node bhNode = bhNodes[levelIdx][k];
      if(prevPersistenceInfo[i][k].dontTraverse != 1) {  // this node's parent intersected
        if(bhNode.isLeaf == true) { // is a leaf node so always calculate force
          //printf("Bhnode mass is %f\n", bhNode.mass);
          if(bhNode.centerOfMassX != points[i].x && bhNode.centerOfMassY != points[i].y) { // check to make sure not computing force against itself
            computedForces[i] += computeObjectsAttractionForce(points[i], bhNode);
          }
        }
        else if(intersectionsOutputData[((i * nodesPerLevel[levelIdx]) + k)] == 0) { // didn't intersect this node so calculate force
          computedForces[i] += computeObjectsAttractionForce(points[i], bhNode); // calculate force

          // set persistence to don't traverse if node has children
          if(bhNode.nw != nullptr) {
            if(bhNode.nw->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.nw), 1));
            if(bhNode.ne->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.ne), 1));
            if(bhNode.sw->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.sw), 1));
            if(bhNode.se->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.se), 1));
          }
        } else { // did intersect this node so don't calculate force
          if(bhNode.nw != nullptr) { // this node has children so set this persistence to true
            if(bhNode.nw->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.nw), 0));
            if(bhNode.ne->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.ne), 0));
            if(bhNode.sw->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.sw), 0));
            if(bhNode.se->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.se), 0));
          }
        }
      } else { // parent already intersected so all children should not be computed
        if(bhNode.nw != nullptr) {
            if(bhNode.nw->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.nw), 1));
            if(bhNode.ne->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.ne), 1));
            if(bhNode.sw->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.sw), 1));
            if(bhNode.se->mass != 0) currentPersistenceInfo.push_back(NodePersistenceInfo(*(bhNode.se), 1));
        }
      }
    }
    prevPersistenceInfo[i].clear();
    prevPersistenceInfo[i].resize(currentPersistenceInfo.size());
    copy(currentPersistenceInfo.begin(), currentPersistenceInfo.end(), prevPersistenceInfo[i].begin());
    currentPersistenceInfo.clear();
  }
}

int main(int ac, char **av) {

  auto total_run_time = chrono::steady_clock::now();


  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################
  BarnesHutTree* tree = new BarnesHutTree(THRESHOLD, gridSize);
  Node* root = new Node(0.f, 0.f, gridSize);

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

  for (int i = 0; i < NUM_POINTS; ++i) {
    Point p;
    p.x = dis(gen);
    p.y = dis(gen);
    p.mass = disMass(gen);
    p.idX = i;
    points.push_back(p);
    //printf("Point # %d has x = %f, y = %f, mass = %f\n", i, p.x, p.y, p.mass);
  }

  OWLBuffer PointsBuffer = owlDeviceBufferCreate(
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());


  LOG("Bulding Tree with # Of Bodies = " << points.size());
  auto start_tree = chrono::steady_clock::now();
  for(const auto& point: points) {
    tree->insertNode(root, point);
  };

  auto end_tree = chrono::steady_clock::now();
  auto elapsed_tree = chrono::duration_cast<chrono::microseconds>(end_tree - start_tree);
  cout << "Barnes Hut Tree Build Time: " << elapsed_tree.count() / 1000000.0 << " seconds."
            << endl;

  //LOG("Size of tree = " << calculateQuadTreeSize(root));
  //tree->printTree(root, 0, "root");

  // Get the device ID
  cudaGetDevice(&deviceID);
  LOG_OK("Device ID: " << deviceID);
  owlGeomTypeSetIntersectProg(SpheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(SpheresGeomType, module, "Spheres");
  owlBuildPrograms(context);

  OWLVarDecl myGlobalsVars[] = {
    {"yIDx", OWL_INT, OWL_OFFSETOF(MyGlobals, yIDx)},
    {"parallelLaunch", OWL_INT, OWL_OFFSETOF(MyGlobals, parallelLaunch)},
    {"outputIntersectionData", OWL_RAW_POINTER, OWL_OFFSETOF(MyGlobals, outputIntersectionData)},
    {"nodesPerLevel", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, nodesPerLevel)},
    {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  //vector<LevelIntersectionInfo> levelIntersectionData;
  vector<OptixTraversableHandle> worlds;
  vector<Sphere> InternalSpheres;
  vector<Node> InternalNodes;
  //LevelIntersectionInfo levelInfo;
  float prevS = gridSize;
  int level = 0;

  // level order traversal of BarnesHutTree
  // Create an empty queue for level order traversal
  queue<Node*> q;
  
  LOG("Bulding OWL Scenes");
  // Enqueue Root and initialize height
  q.push(root);
  auto start_b = chrono::steady_clock::now();
  while (q.empty() == false) {
      // Print front of queue and remove it from queue
      Node* node = q.front();
      if((node->s != prevS)) {
        if(!InternalSpheres.empty()) {
          worlds.push_back(createSceneGivenGeometries(InternalSpheres, (gridSize / THRESHOLD)));
          // if(!offsetPerLevel.empty()) {
          //   offsetPerLevel.push_back((offsetPerLevel.back() + (NUM_POINTS * nodesPerLevel.back())));
          // } else {
          //   offsetPerLevel.push_back(0);
          // }
          nodesPerLevel.push_back(InternalSpheres.size());
          if(maxNodesPerLevel < InternalSpheres.size()) {
            maxNodesPerLevel = InternalSpheres.size();
          }
          bhNodes.push_back(InternalNodes);
        } else {
          LOG_OK("Spheres r empty!");
        }
        InternalSpheres.clear();
        InternalNodes.clear();
        prevS = node->s;
        gridSize = node->s;
        level += 1;
      } else {
        //LOG_OK("HITS THIS!");
      }
      if(node->s == gridSize) {
        if(node->mass != 0.0f) {
          if(node->ne == nullptr) {
            node->isLeaf = true;
          } else {
            node->isLeaf = false;
          }
          InternalNodes.push_back((*node));
          totalNumNodes += 1;
          InternalSpheres.push_back(Sphere{vec3f{node->centerOfMassX, node->centerOfMassY, 0}, node->mass});
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

  // last level gets missed in while loop so this is to account for that
  if(!InternalSpheres.empty()) {
    worlds.push_back(createSceneGivenGeometries(InternalSpheres, (gridSize / THRESHOLD)));
    // if(!offsetPerLevel.empty()) {
    //   offsetPerLevel.push_back((offsetPerLevel.back() + (NUM_POINTS * nodesPerLevel.back())));
    // } else {
    //   offsetPerLevel.push_back(0);
    // }
    nodesPerLevel.push_back(InternalSpheres.size());
    if(maxNodesPerLevel < InternalSpheres.size()) {
      maxNodesPerLevel = InternalSpheres.size();
    }
    bhNodes.push_back(InternalNodes);
  }

  long int numElementsInOutputIntersectionData = (NUM_POINTS * maxNodesPerLevel);
  printf("Size of outputIntersectionData is = %lu bytes.\n", numElementsInOutputIntersectionData*sizeof(uint8_t));

  auto end_b = chrono::steady_clock::now();
  auto elapsed_b = chrono::duration_cast<chrono::microseconds>(end_b - start_b);
  cout << "OWL Scenes Build time: " << elapsed_b.count() / 1000000.0 << " seconds."
            << endl;

  // create space for output intersection info on device using unified memory
  OWLBuffer NodesPerLevelBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(nodesPerLevel[0]), nodesPerLevel.size(), nodesPerLevel.data());
  
  // OWLBuffer OffsetPerLevelBuffer = owlManagedMemoryBufferCreate( 
  //    context, OWL_USER_TYPE(offsetPerLevel[0]), offsetPerLevel.size(), offsetPerLevel.data());
  
  hostOutputIntersectionData = new uint8_t[numElementsInOutputIntersectionData]();
  //zeroOutputIntersectionData = new uint8_t[numElementsInOutputIntersectionData]();

  cudaError_t levelCudaStatus = cudaMalloc(&deviceOutputIntersectionData, numElementsInOutputIntersectionData*sizeof(uint8_t));
  if(levelCudaStatus != cudaSuccess) printf("cudaMallocManaged failed: %s\n", cudaGetErrorString(levelCudaStatus));
  levelCudaStatus = cudaMemset(deviceOutputIntersectionData, 0, numElementsInOutputIntersectionData*sizeof(uint8_t));
  //levelCudaStatus = cudaMemcpy(deviceOutputIntersectionData, zeroOutputIntersectionData, numElementsInOutputIntersectionData*sizeof(uint8_t), cudaMemcpyHostToDevice);
  if(levelCudaStatus != cudaSuccess) printf("cudaMemcpy failed: %s\n", cudaGetErrorString(levelCudaStatus));

  for(int i = 0; i < worlds.size(); i++) {
    printf("For level %d there are %lu nodes\n", i, nodesPerLevel[i]);
    //printf("For level %d the offset is %lu.\n", i, offsetPerLevel[i]);
    // cudaError_t pointCudaStatus = cudaMalloc(&(deviceOutputIntersectionData[i].pointIntersectionInfo), (NUM_POINTS * nodesPerLevel[i]) * sizeof(uint8_t));
    // if(pointCudaStatus != cudaSuccess) printf("cudaMallocManaged failed at level %d: %s\n", i, cudaGetErrorString(pointCudaStatus));
    // pointCudaStatus = cudaMemset(deviceOutputIntersectionData[i].pointIntersectionInfo, 0, (NUM_POINTS * nodesPerLevel[i]) * sizeof(uint8_t));
    // if(pointCudaStatus != cudaSuccess) printf("cudaMemset failed at level %d: %s\n", i, cudaGetErrorString(pointCudaStatus));
  }

  OWLBuffer WorldsBuffer = owlDeviceBufferCreate(
        context, OWL_USER_TYPE(worlds[0]), worlds.size(), worlds.data());
  
  LOG_OK("Number of Worlds: " << worlds.size());
  LOG_OK("Total number of non-empty BH Nodes: " << totalNumNodes);
  
  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
      //{"internalSpheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, internalSpheres)},
      {"points", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, points)},
      {"worlds", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, worlds)},
      {/* sentinel to mark end of list */}};

  // ........... create object  ............................
  OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                                     sizeof(RayGenData), rayGenVars, -1);\
  
                           

  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen, "points", PointsBuffer);
  owlRayGenSetBuffer(rayGen, "worlds", WorldsBuffer);

  // programs have been built before, but have to rebuild raygen and
  // miss progs

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  owlParamsSetPointer(lp, "outputIntersectionData", deviceOutputIntersectionData);
  owlParamsSetBuffer(lp, "nodesPerLevel", NodesPerLevelBuffer);
  //owlParamsSetBuffer(lp, "offsetPerLevel", OffsetPerLevelBuffer);

  // // ##################################################################
  // // Start Ray Tracing Parallel launch
  // // ##################################################################
  auto start1 = std::chrono::steady_clock::now();
  owlParamsSet1i(lp, "parallelLaunch", 1);
  //owlLaunch2D(rayGen, points.size(), worlds.size() - 1, lp);

  auto end1 = std::chrono::steady_clock::now();
  auto elapsed1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cout << "Intersections time for parallel launch: " << elapsed1.count() / 1000000.0
            << " seconds." << std::endl;
  
  // ##################################################################
  // Start Ray Tracing Series launch
  // ##################################################################

  // intersection computation info initialization
  for(int i = 0; i < NUM_POINTS; i++) {
    vector<NodePersistenceInfo> prevPersistenceInfoForAPoint(4, NodePersistenceInfo());
    prevPersistenceInfo.push_back(prevPersistenceInfoForAPoint);
  }

  auto start2 = chrono::steady_clock::now();
  owlParamsSet1i(lp, "parallelLaunch", 0);
  for(int l = 1; l < worlds.size(); l++) {
    owlParamsSet1i(lp, "yIDx", l);
    owlLaunch2D(rayGen, points.size(), 1, lp);

    // calculate forces here
    // cudaError_t intersectionsStatus = cudaMemcpy(hostOutputIntersectionData, deviceOutputIntersectionData, numElementsInOutputIntersectionData*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // if(intersectionsStatus != cudaSuccess) printf("cudaMemcpy failed: %s\n", cudaGetErrorString(intersectionsStatus));
    //computeForces(hostOutputIntersectionData, points, l);

    // set each element in deviceOutputIntersectionData to 0
    // intersectionsStatus = cudaMemset(deviceOutputIntersectionData, 0, numElementsInOutputIntersectionData*sizeof(uint8_t));
    // // intersectionsStatus = cudaMemcpy(deviceOutputIntersectionData, zeroOutputIntersectionData, numElementsInOutputIntersectionData*sizeof(uint8_t), cudaMemcpyHostToDevice);
    // if(intersectionsStatus != cudaSuccess) printf("cudaMemcpy failed: %s\n", cudaGetErrorString(intersectionsStatus));
  }
  auto end2 = chrono::steady_clock::now();
  auto elapsed2 =
      chrono::duration_cast<chrono::microseconds>(end2 - start2);
  cout << "Intersections + calculation time for series launch: " << elapsed2.count() / 1000000.0
            << " seconds." << endl;

  // ##################################################################
  // Output Force Computations
  // ##################################################################
  // compute real forces using cpu BH traversal
  auto cpuforcesstart = chrono::steady_clock::now();
  tree->computeForces(root, points, cpuComputedForces);
  auto cpuforcesend = chrono::steady_clock::now();
  auto cpuforceselapsed = chrono::duration_cast<chrono::microseconds>(cpuforcesend - cpuforcesstart);
  cout << "Cpu forces calculation time: " << cpuforceselapsed.count() / 1000000.0
            << " seconds." << endl;


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

  //tree->printTree(root, 0, "root");
  //auto cpuforcesstart = chrono::steady_clock::now();
  //auto cpuforcesend = chrono::steady_clock::now();
  // auto cpuforceselapsed =
  //     chrono::duration_cast<chrono::microseconds>(cpuforcesend - cpuforcesstart);
  // cout << "Cpu forces calculation time: " << cpuforceselapsed.count() / 1000000.0
  //           << " seconds." << endl;

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  auto total_run_time_end = chrono::steady_clock::now();
  auto elapsed_run_time_end = chrono::duration_cast<chrono::microseconds>(total_run_time_end - total_run_time);
  cout << "Total run time is: " << elapsed_run_time_end.count() / 1000000.0 << " seconds."
            << endl;
  // free memory
  cudaFree(deviceOutputIntersectionData);
  delete[] hostOutputIntersectionData;

  // destory owl stuff
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
}