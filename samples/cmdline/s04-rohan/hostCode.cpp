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
#include "optix.h"
#include <owl/DeviceMemory.h>
// our device-side data structures
#include "GeomTypes.h"
//#include "owlViewer/OWLViewer.h"

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
#include <limits>
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
#define INTERSECTIONS_SIZE 400
#define MIN_DISTANCE 2.0f
#define TRIANGLEX_THRESHOLD 50000.0f // this has to be greater than the GRID_SIZE in barnesHutTree.h

// random init
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dis(-5000.0f, 5000.0f);  // Range for X and Y coordinates
uniform_real_distribution<float> disMass(100.0f, 2000.0f);  // Range mass 

// global variables
int deviceID;
float gridSize = GRID_SIZE;
vector<long int> nodesPerLevel;
long int maxNodesPerLevel = 0;
ProfileStatistics *profileStats = new ProfileStatistics();
u_int *deviceOutputIntersectionData;

// force calculation global variables
vector<CustomRay> primaryLaunchRays(NUM_POINTS);
vector<CustomRay> orderedPrimaryLaunchRays(NUM_POINTS);
vector<deviceBhNode> deviceBhNodes;
deviceBhNode *deviceBhNodesPointer;
Point *devicePoints;
int totalNumNodes = 0;
float minNodeSValue = GRID_SIZE + 1.0f;
vector<float> computedForces(NUM_POINTS, 0.0f);
vector<float> maxForces(NUM_POINTS, 0.0f);
vector<float> cpuComputedForces(NUM_POINTS, 0.0f);
vector<IntersectionResult> intersectionResults(INTERSECTIONS_SIZE);

vector<vec3f> vertices;
vector<vec3i> indices;
vector<Point> points;
vector<Node*> dfsBHNodes;

OWLContext context = owlContextCreate(nullptr, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

float euclideanDistance(const Point& p1, const Point& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void dfsTreeSetup() {
  float prevS = gridSize;
  float nextGridSize = gridSize / 2.0;
  float level = 0.0f;
  long int currentNodesPerLevel = 0;
  int currentIndex = 0;
  int indIndex = 0;
  float triangleXLocation = 0.0f;
  //float rayOriginXLocation = 0;
  int primIDIndex = 1;
  int index = 0;
  vertices.resize(dfsBHNodes.size() * 3);
  indices.resize(dfsBHNodes.size());
  deviceBhNodes.resize(dfsBHNodes.size());

  // level order traversal of BarnesHutTree
  // Create an empty queue for level order traversal
  
  LOG("Bulding OWL Scenes");
  // Enqueue Root and initialize height
  while (index < dfsBHNodes.size()) {
      // Print front of queue and remove it from queue
      Node* node = dfsBHNodes[index];

      if(node->ne == nullptr) {
        node->isLeaf = true;
      } else {
        node->isLeaf = false;
      }
      // add triangle to scene corresponding to barnes hut node
      // if(node->mass == static_cast<float>(2505.400879)) {
      //   printf("Triangle loc prev %.9f\n", static_cast<float>(triangleXLocation));
      // }
      //triangleXLocation = triangleXLocation + FloatType(node->s);
      triangleXLocation += node->s;
      if(node->s < minNodeSValue) minNodeSValue = node->s;
      if(triangleXLocation >= TRIANGLEX_THRESHOLD) {
        level += 3.0f;
        triangleXLocation = node->s;
      }
      // if(node->mass == static_cast<float>(2505.400879)) {
      //   printf("Node-s is %.9f\n",  node->s);
      //   // printf("Node-s as double is %.9f\n", s);
      //   //float triangleXDecimal = triangleXLocation - std::floor(triangleXLocation);
      //   //triangleXDecimal = std::round(triangleXDecimal * 1000000.0) / 1000000.0;
      //   //float floatTriangleXLocation = static_cast<float>(triangleXLocation);
      //   //floatTriangleXLocation = std::floor(floatTriangleXLocation) + triangleXDecimal;
      //   printf("Triangle Location: %.9f\n", static_cast<float>(triangleXLocation));
      // }
      
      float triangleXLocationCasted = static_cast<float>(triangleXLocation); // super costly operation
      //float triangleXLocationCasted = 0.0f;
      vertices[currentIndex] = vec3f{triangleXLocationCasted, 0.0f + level, -0.5f};
      vertices[currentIndex+1] = vec3f{triangleXLocationCasted, -0.5f + level, 0.5f};
      vertices[currentIndex+2] = vec3f{triangleXLocationCasted, 0.5f + level, 0.5f};
      indices[indIndex] = vec3i{currentIndex, currentIndex+1, currentIndex+2};
      currentIndex += 3;

      // add device bhNode to vector array
      deviceBhNode currentBhNode;
      currentBhNode.mass = node->mass;
      currentBhNode.s = node->s;
      currentBhNode.centerOfMassX = node->centerOfMassX;
      currentBhNode.centerOfMassY = node->centerOfMassY;
      currentBhNode.isLeaf = node->isLeaf ? 1 : 0;
      currentBhNode.nextRayLocation = vec3f{triangleXLocationCasted, level, 0.0f};
      currentBhNode.nextPrimId = primIDIndex;

      deviceBhNodes[indIndex] = currentBhNode;
      indIndex += 1;
      primIDIndex++;
      // update counters
      totalNumNodes++; //currentNodesPerLevel++;
      
      index++;
  }
}

void orderPointsDFS() {
  int primaryRayIndex = 0;
  for(int i = 0; i < dfsBHNodes.size(); i++) {
    //printf("deviceBhNodes index: %d | nextPrimId %d\n", i, deviceBhNodes[i].nextPrimId);
    if(dfsBHNodes[i]->ne == nullptr) {
      orderedPrimaryLaunchRays[primaryRayIndex].pointID = dfsBHNodes[i]->pointID;
      orderedPrimaryLaunchRays[primaryRayIndex].primID = 0;
      orderedPrimaryLaunchRays[primaryRayIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
      primaryRayIndex++;
    }
  }
}


void treeToDFSArray(Node *node, std::map<Node*, int>& addressToIndex, int& dfsIndex) {
  if(node != nullptr) {
    dfsBHNodes.push_back(node);
    addressToIndex[node] = dfsIndex;
    dfsIndex++;

    if(node->nw != nullptr) {
      if(node->nw->mass != 0.0f) treeToDFSArray(node->nw, addressToIndex, dfsIndex);
      if(node->ne->mass != 0.0f) treeToDFSArray(node->ne, addressToIndex, dfsIndex);
      if(node->sw->mass != 0.0f) treeToDFSArray(node->sw, addressToIndex, dfsIndex);
      if(node->se->mass != 0.0f) treeToDFSArray(node->se, addressToIndex, dfsIndex);
    }
  }
}

void installAutoRopes(Node* root, std::map<Node*, int> addressToIndex) {
    std::stack<Node*> ropeStack;
    ropeStack.push(root);
    int initial = 0;
    int maxStackSize = 0;
    int iterations = 0;

    while (!ropeStack.empty()) {
      Node* currentNode = ropeStack.top();
      if(ropeStack.size() > maxStackSize) maxStackSize = ropeStack.size();
      //printf("Stack Node ---> ");
      //std::cout << "Node: Mass = " << currentNode->mass << ", Center of Mass = (" << currentNode->centerOfMassX << ", " << currentNode->centerOfMassY << "), quadrant = (" << currentNode->quadrantX << ", " << currentNode->quadrantY << ")" << "\n";
      ropeStack.pop();

      int dfsIndex = addressToIndex[currentNode];
      if(initial) {
        //printf("Roping to ---> ");
        if(ropeStack.size() != 0) {
          Node* autoRopeNode = ropeStack.top();
          int ropeIndex = addressToIndex[autoRopeNode];
          //std::cout << "Node: Mass = " << autoRopeNode->mass << ", Center of Mass = (" << autoRopeNode->centerOfMassX << ", " << autoRopeNode->centerOfMassY << "), quadrant = (" << autoRopeNode->quadrantX << ", " << autoRopeNode->quadrantY << ")" << "\n";
          deviceBhNodes[dfsIndex].autoRopeRayLocation = deviceBhNodes[ropeIndex-1].nextRayLocation;
          deviceBhNodes[dfsIndex].autoRopePrimId = deviceBhNodes[ropeIndex-1].nextPrimId;
        } else {
          deviceBhNodes[dfsIndex].autoRopeRayLocation = vec3f{0.0f, 0.0f, 0.0f};
          deviceBhNodes[dfsIndex].autoRopePrimId = 0;
          //printf("END!\n");
        }
      } else {
        initial = 1;
      } 

      // Enqueue child nodes in the desired order
      if (currentNode->se && currentNode->se->mass != 0.0f) ropeStack.push(currentNode->se);
      if (currentNode->sw && currentNode->sw->mass != 0.0f) ropeStack.push(currentNode->sw);
      if (currentNode->ne && currentNode->ne->mass != 0.0f) ropeStack.push(currentNode->ne);
      if (currentNode->nw && currentNode->nw->mass != 0.0f) ropeStack.push(currentNode->nw);
      iterations++;
    }

    printf("Max stack size: %d\n", maxStackSize);
    printf("Iterations: %d\n", iterations);

}

int main(int ac, char **av) {
  auto total_run_time = chrono::steady_clock::now();

  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################
  BarnesHutTree* tree = new BarnesHutTree(THRESHOLD, gridSize);
  Node* root = new Node(0.f, 0.f, gridSize);
  
  FILE *outFile = fopen("../points.txt", "w");
  if (!outFile) {
    std::cerr << "Error opening file for writing." << std::endl;
    return 1;
  }

  fprintf(outFile, "%d\n", NUM_POINTS);
  fprintf(outFile, "%d\n", NUM_STEPS);
  fprintf(outFile, "%f\n", (0.025));
  fprintf(outFile, "%f\n", (0.05));
  fprintf(outFile, "%f\n", THRESHOLD);

  int numPointsSoFar = 0;
  while (numPointsSoFar < NUM_POINTS) {
    Point p;
    p.x = dis(gen);
    p.y = dis(gen);
    p.z = 0.0f;
    p.vel_x = 0.0f;
    p.vel_y = 0.0f;
    p.vel_z = 0.0f;
    p.mass = disMass(gen);
    p.idX = numPointsSoFar;

    fprintf(outFile, "%f %f %f %f %f %f %f\n", p.mass, p.x, p.y, p.z, p.vel_x, p.vel_y, p.vel_z);
    //outFile.write(reinterpret_cast<char*>(&p), sizeof(Point));
    points.push_back(p);
    //printf("Point # %d has x = %f, y = %f, mass = %f\n", i, p.x, p.y, p.mass);
    primaryLaunchRays[numPointsSoFar].pointID = numPointsSoFar;
    primaryLaunchRays[numPointsSoFar].primID = 0;
    primaryLaunchRays[numPointsSoFar].orgin = vec3f(0.0f, 0.0f, 0.0f);
    numPointsSoFar++;
  }
  fclose(outFile);

  // FILE *inFile = fopen("../points.txt", "r");
  // if (!inFile) {
  //     std::cerr << "Error opening file for reading." << std::endl;
  //     return 1;
  // }

  // float randomStuff;

  // for(int i = 0; i < 5; i++) {
  //   fscanf(inFile, "%f\n", &randomStuff);
  //   printf("Read %f\n", randomStuff);
  // }

  // int launchIndex = 0;
  // float x, y,z, mass;
  // vec3f velRead;
  //   // Read three floats from each line until the end of the file
  // while (fscanf(inFile, "%f %f %f %f %f %f %f", &mass, &x, &y, &z, &(velRead.x), &(velRead.y), &(velRead.z)) == 7) {
  //     //Process the floats as needed
  //     Point point;
  //     point.x = x;
  //     point.y = y;
  //     point.z = 0.0f;
  //     point.vel_x = velRead.x;
  //     point.vel_y = velRead.u;
  //     point.vel_z = velRead.z;
  //     point.mass = mass;
  //     point.idX = launchIndex;

  //     if(launchIndex == 0) {
  //       printf("Read: mass=%f, x=%f, y=%f, z=%f, vel_x=%f, vel_y=%f, vel_z=%f\n", mass, x, y, z, velRead.x, velRead.y, velRead.z);
  //     }

  //     points.push_back(point);
  //     primaryLaunchRays[launchIndex].pointID = launchIndex;
  //     primaryLaunchRays[launchIndex].primID = 0;
  //     primaryLaunchRays[launchIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
  //     launchIndex++;
  // }
  // fclose(inFile);

  OWLBuffer PointsBuffer = owlDeviceBufferCreate(
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());



  LOG("Bulding Tree with # Of Bodies = " << points.size());
  auto tree_build_time_start = chrono::steady_clock::now();
  for(const auto& point: points) {
    tree->insertNode(root, point);
  };
  auto tree_build_time_end = chrono::steady_clock::now();
  profileStats->treeBuildTime += chrono::duration_cast<chrono::microseconds>(tree_build_time_end - tree_build_time_start);

  auto tree_to_dfs_time_start = chrono::steady_clock::now();
  std::map<Node*, int> addressToIndex;
  int dfsIndex = 0;
  treeToDFSArray(root, addressToIndex, dfsIndex);
  printf("Number of dfsNodes %lu\n",dfsBHNodes.size());
  auto tree_to_dfs_time_end = chrono::steady_clock::now();
  profileStats->treeToDFSTime += chrono::duration_cast<chrono::microseconds>(tree_to_dfs_time_end - tree_to_dfs_time_start);

  // order points in dfs order
  orderPointsDFS();

  OWLBuffer primaryLaunchRaysBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(orderedPrimaryLaunchRays[0]),
                            orderedPrimaryLaunchRays.size(),orderedPrimaryLaunchRays.data());

  //tree->printTree(root, 0, "root");

  // Get the device ID
  cudaGetDevice(&deviceID);
  LOG_OK("Device ID: " << deviceID);

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  auto scene_build_time_start = chrono::steady_clock::now();
  auto test_time_start = chrono::steady_clock::now();
  dfsTreeSetup();
  auto test_time_end = chrono::steady_clock::now();
  profileStats->createSceneTime += chrono::duration_cast<chrono::microseconds>(test_time_end - test_time_start);
  // for(int i = 0; i < deviceBhNodes.size(); i++) {
  //   //printf("deviceBhNodes index: %d | nextPrimId %d\n", i, deviceBhNodes[i].nextPrimId);
  // }

  auto auto_ropes_start = chrono::steady_clock::now();
  installAutoRopes(root, addressToIndex);
  auto auto_ropes_end = chrono::steady_clock::now();
  profileStats->installAutoRopesTime += chrono::duration_cast<chrono::microseconds>(auto_ropes_end - auto_ropes_start);

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

  auto scene_build_time_end = chrono::steady_clock::now();
  profileStats->sceneBuildTime += chrono::duration_cast<chrono::microseconds>(scene_build_time_end - scene_build_time_start);
  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  auto intersections_setup_time_start = chrono::steady_clock::now();
  OWLVarDecl missProgVars[]
    = {
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);


  OWLVarDecl myGlobalsVars[] = {
    {"deviceBhNodes", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, deviceBhNodes)},
    {"devicePoints", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, devicePoints)},
    {"numPrims", OWL_INT, OWL_OFFSETOF(MyGlobals, numPrims)},
    {"computedForces", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, computedForces)},
    {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);


  OWLBuffer DevicePointsBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());
  owlParamsSetBuffer(lp, "devicePoints", DevicePointsBuffer);

  OWLBuffer DeviceBhNodesBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(deviceBhNodes[0]), deviceBhNodes.size(), deviceBhNodes.data());
  owlParamsSetBuffer(lp, "deviceBhNodes", DeviceBhNodesBuffer);

  OWLBuffer ComputedForcesBuffer = owlManagedMemoryBufferCreate( 
     context, OWL_FLOAT, NUM_POINTS, nullptr);
  owlParamsSetBuffer(lp, "computedForces", ComputedForcesBuffer);

  owlParamsSet1i(lp, "numPrims", deviceBhNodes.size());
  

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


  printf("Size of malloc for bhNodes is %.2f gb.\n", (deviceBhNodes.size() * sizeof(deviceBhNode))/1000000000.0);
  printf("Size of malloc for points is %.2f gb.\n", (points.size() * sizeof(Point))/1000000000.0);
  printf("Minimum node-s value %f\n", minNodeSValue);
  
  auto intersections_setup_time_end = chrono::steady_clock::now();
  profileStats->intersectionsSetupTime += chrono::duration_cast<chrono::microseconds>(intersections_setup_time_end - intersections_setup_time_start);

  owlContextPrintSER(context);

  // ##################################################################
  // Start Ray Tracing Parallel launch
  // ##################################################################
  auto start1 = std::chrono::steady_clock::now();
  owlLaunch2D(rayGen, NUM_POINTS, 1, lp);
  const float *rtComputedForces = (const float *)owlBufferGetPointer(ComputedForcesBuffer,0);
  auto end1 = std::chrono::steady_clock::now();
  profileStats->forceCalculationTime = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

  // ##################################################################
  // Output Force Computations
  // ##################################################################
  // compute real forces using cpu BH traversal
  printf("CPU OUTPUT!!!\n");
  printf("----------------------------------------\n");
  auto cpu_forces_start_time = chrono::steady_clock::now();
  tree->computeForces(root, points, cpuComputedForces);
  auto cpu_forces_end_time = chrono::steady_clock::now();
  profileStats->cpuForceCalculationTime += chrono::duration_cast<chrono::microseconds>(cpu_forces_end_time - cpu_forces_start_time);
  printf("----------------------------------------\n");

  int pointsFailing = 0;
  for(int i = 0; i < NUM_POINTS; i++) {
    float percent_error = (abs((rtComputedForces[i] - cpuComputedForces[i])) / cpuComputedForces[i]) * 100.0f;
    if(percent_error > 2.5f) {
      // LOG_OK("++++++++++++++++++++++++");
      // LOG_OK("POINT #" << i << ", (" << points[i].x << ", " << points[i].y << ") , HAS ERROR OF " << percent_error << "%");
      // LOG_OK("++++++++++++++++++++++++");
      // printf("RT force = %f\n", rtComputedForces[i]);
      // printf("CPU force = %f\n", cpuComputedForces[i]);
      // LOG_OK("++++++++++++++++++++++++");
      // printf("\n");
      pointsFailing++;
    }
  }
  printf("Points failing percent error: %d\n", pointsFailing);

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
  std::cout << "Tree to DFS time: " << profileStats->treeToDFSTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Create Scene Time: " << profileStats->createSceneTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Install AutoRopes time: " << profileStats->installAutoRopesTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Intersections setup time: " << profileStats->intersectionsSetupTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "RT Cores Force Calculations time: " << profileStats->forceCalculationTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "CPU Force Calculations time: " << profileStats->cpuForceCalculationTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Total Program time: " << profileStats->totalProgramTime.count() / 1000000.0 << " seconds." << std::endl;
  printf("--------------------------------------------------------------\n");

  // destory owl stuff
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
}