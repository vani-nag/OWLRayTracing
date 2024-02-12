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
#include "kernels.h"
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
#include <unordered_map> 

#define LOG(message)                                            \
  cout << OWL_TERMINAL_BLUE;                               \
  cout << "#owl.sample(main): " << message << endl;    \
  cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  cout << "#owl.sample(main): " << message << endl;    \
  cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

#define TRIANGLEX_THRESHOLD 50000.0f // this has to be greater than the GRID_SIZE in barnesHutTree.h

// random init
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dis(GRID_SIZE/-2.0f, GRID_SIZE/2.0f);  // Range for X and Y coordinates
uniform_real_distribution<float> disMass(10.0f, 2000.0f);  // Range mass 

// global variables
float gridSize = GRID_SIZE;
ProfileStatistics *profileStats = new ProfileStatistics();

// force calculation global variables
vector<CustomRay> primaryLaunchRays(NUM_POINTS);
vector<CustomRay> orderedPrimaryLaunchRays(NUM_POINTS);
vector<deviceBhNode> deviceBhNodes;
Point *devicePoints;
float minNodeSValue = GRID_SIZE + 1.0f;
vector<float> computedForces(NUM_POINTS, 0.0f);
vector<float> cpuComputedForces(NUM_POINTS, 0.0f);

vector<vec3f> vertices;
vector<vec3i> indices;
vector<Point> points;
vector<Node*> dfsBHNodes;

OWLContext context = owlContextCreate(nullptr, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

void treeToDFSArray(Node* root) {
  std::stack<Node*, std::vector<Node*>> nodeStack;
  int dfsIndex = 0;
  float triangleXLocation = 0.0f;
  float level = 0.0f;
  int currentIndex = 0;
  int primIDIndex = 1;
  //vertices.reserve((NUM_POINTS * 3) * 3);
  //indices.reserve(NUM_POINTS * 3);
  //deviceBhNodes.reserve(NUM_POINTS * 3);
  //dfsBHNodes.reserve(NUM_POINTS * 3);
  // chrono::steady_clock::time_point clock_begin;
  // chrono::steady_clock::time_point clock_end;
  // chrono::steady_clock::duration time_span;
  double nseconds = 0.0f;

  nodeStack.push(root);
  while (!nodeStack.empty()) {
    Node* currentNode = nodeStack.top();
    nodeStack.pop();
    if(currentNode != nullptr) {
      // determine if leaf node
      // if(currentNode->children[0] == nullptr) {
      //   currentNode->type = bhLeafNode;
      // } else {
      //   currentNode->type = bhNonLeafNode;
      // }

      // calculate triangle location
      triangleXLocation += std::sqrt(currentNode->s);
      if(currentNode->s < minNodeSValue) minNodeSValue = currentNode->s;
      if(triangleXLocation >= TRIANGLEX_THRESHOLD) {
        level += 3.0f;
        triangleXLocation = std::sqrt(currentNode->s);
      }

      // add triangle to scene corresponding to barnes hut node
      vertices.push_back({triangleXLocation, 0.0f + level, -0.5f});
      vertices.push_back({triangleXLocation, -0.5f + level, 0.5f});
      vertices.push_back({triangleXLocation, 0.5f + level, 0.5f});
      indices.push_back({currentIndex, currentIndex+1, currentIndex+2});
      currentIndex += 3;

      // create device bhNode
      deviceBhNode currentBhNode;
      currentBhNode.mass = currentNode->mass;
      currentBhNode.s = currentNode->s;
      currentBhNode.centerOfMassX = currentNode->cofm.x;
      currentBhNode.centerOfMassY = currentNode->cofm.y;
      currentBhNode.centerOfMassZ = currentNode->cofm.z;
      currentBhNode.isLeaf = (currentNode->type == bhLeafNode) ? 1 : 0;
      currentBhNode.nextRayLocation = vec3f{triangleXLocation, level, 0.0f};
      currentBhNode.nextPrimId = primIDIndex;

      deviceBhNodes.push_back(currentBhNode);
      primIDIndex++;

      // add bhNode to DFS array
      currentNode->dfsIndex = dfsIndex;
      dfsBHNodes.push_back(currentNode);
      dfsIndex++;

      // Enqueue child nodes in the desired order
      //clock_begin = chrono::steady_clock::now();
      for(int i = 7; i >= 0; i--) {
        if(currentNode->children[i] != nullptr && currentNode->children[i]->mass != 0.0f) {
          nodeStack.push(currentNode->children[i]);
        }
      }

      //clock_end = chrono::steady_clock::now();
      //time_span = clock_end - clock_begin;
      //nseconds += double(time_span.count()) * chrono::steady_clock::period::num / chrono::steady_clock::period::den;
    }
  }

  std::cout << "Profile Time in Kernel: " << nseconds << std::endl;
}

void orderPointsDFS() {
  int primaryRayIndex = 0;
  for(int i = 0; i < dfsBHNodes.size(); i++) {
    //printf("deviceBhNodes index: %d | nextPrimId %d\n", i, deviceBhNodes[i].nextPrimId);
    for(int j = 0; j < 8; j++) {
      if(dfsBHNodes[i]->children[j] != nullptr) {
        orderedPrimaryLaunchRays[primaryRayIndex].pointID = dfsBHNodes[i]->pointID;
        orderedPrimaryLaunchRays[primaryRayIndex].primID = 0;
        orderedPrimaryLaunchRays[primaryRayIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
        primaryRayIndex++;
      }
    }
  }
}

void installAutoRopes(Node* root) {
    std::stack<Node*> ropeStack;
    ropeStack.push(root);
    int initial = 0;
    int maxStackSize = 0;
    int iterations = 0;

    while (!ropeStack.empty()) {
      Node* currentNode = ropeStack.top();
      if(ropeStack.size() > maxStackSize) maxStackSize = ropeStack.size();
      ropeStack.pop();

      int dfsIndex = currentNode->dfsIndex;
      if(initial) {
        if(ropeStack.size() != 0) {
          Node* autoRopeNode = ropeStack.top();
          int ropeIndex = autoRopeNode->dfsIndex;
          deviceBhNodes[dfsIndex].autoRopeRayLocation = deviceBhNodes[ropeIndex-1].nextRayLocation;
          deviceBhNodes[dfsIndex].autoRopePrimId = deviceBhNodes[ropeIndex-1].nextPrimId;
        } else {
          deviceBhNodes[dfsIndex].autoRopeRayLocation = vec3f{0.0f, 0.0f, 0.0f};
          deviceBhNodes[dfsIndex].autoRopePrimId = 0;
        }
      } else {
        initial = 1;
      } 

      // Enqueue child nodes in the desired order
      for(int i = 7; i >= 0; i--) {
        if(currentNode->children[i] != nullptr && currentNode->children[i]->mass != 0.0f) {
          ropeStack.push(currentNode->children[i]);
        }
      }
      iterations++;
    }

    printf("Max stack size: %d\n", maxStackSize);
    printf("Iterations: %d\n", iterations);
}

int main(int ac, char **av) {

  if(ac == 3) {  
    if(std::string(av[1]) == "new") {
      FILE *outFile = fopen(av[2], "w");
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
        p.pos.x = dis(gen);
        p.pos.y = dis(gen);
        p.pos.z = dis(gen);
        p.vel.x = 0.0f;
        p.vel.y= 0.0f;
        p.vel.z = 0.0f;
        p.mass = disMass(gen);
        p.idX = numPointsSoFar;

        fprintf(outFile, "%f %f %f %f %f %f %f\n", p.mass, p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z);
        //outFile.write(reinterpret_cast<char*>(&p), sizeof(Point));
        points.push_back(p);
        //printf("Point # %d has x = %f, y = %f, mass = %f\n", i, p.x, p.y, p.mass);
        primaryLaunchRays[numPointsSoFar].pointID = numPointsSoFar;
        primaryLaunchRays[numPointsSoFar].primID = 0;
        primaryLaunchRays[numPointsSoFar].orgin = vec3f(0.0f, 0.0f, 0.0f);
        numPointsSoFar++;
      }
      fclose(outFile);
    } else if(std::string(av[1]) == "treelogy") {
      FILE *inFile = fopen(av[2], "r");
      if (!inFile) {
          std::cerr << "Error opening file for reading." << std::endl;
          return 1;
      }
      float randomStuff;

      for(int i = 0; i < 5; i++) {
        fscanf(inFile, "%f\n", &randomStuff);
        printf("Read %f\n", randomStuff);
      }

      int launchIndex = 0;
      float x, y,z, mass;
      vec3f velRead;
        // Read three floats from each line until the end of the file
      while (fscanf(inFile, "%f %f %f %f %f %f %f", &mass, &x, &y, &z, &(velRead.x), &(velRead.y), &(velRead.z)) == 7) {
          //Process the floats as needed
          Point point;
          point.pos.x = x;
          point.pos.y = y;
          point.pos.z = z;
          point.vel.x = velRead.x;
          point.vel.y = velRead.u;
          point.vel.z = velRead.z;
          point.mass = mass;
          point.idX = launchIndex;

          points.push_back(point);
          primaryLaunchRays[launchIndex].pointID = launchIndex;
          primaryLaunchRays[launchIndex].primID = 0;
          primaryLaunchRays[launchIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
          launchIndex++;
      }
      fclose(inFile);
    } else {
      std::cout << "Unsupported filetype format: " << av[1] << std::endl;
      return 1; // Exit with an error code
    }
  } else {
    // Incorrect command-line arguments provided
    std::cout << "Use Existing Dataset: " << av[0] << " <treelogy/tipsy> <filepath>" << std::endl;
    std::cout << "Generate Synthetic Dataset: " << av[0] << " <new> <filepath>" << std::endl;
    return 1; // Exit with an error code
  }
  auto total_run_time = chrono::steady_clock::now();
  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################
  BarnesHutTree* tree = new BarnesHutTree(THRESHOLD, gridSize);
  Node* root = new Node(points[0].pos.x, points[0].pos.x, points[0].pos.x, gridSize, points[0].idX);
  root->mass = points[0].mass;
  printf("Node: Mass = %f, Center of Mass = (%f, %f, %f)\n", points[0].mass, points[0].pos.x, points[0].pos.y, points[0].pos.z);

  OWLBuffer PointsBuffer = owlDeviceBufferCreate(
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());

  LOG("Bulding Tree with # Of Bodies = " << points.size());
  auto tree_build_time_start = chrono::steady_clock::now();
  int num = 0;
  for(int i = 1; i < points.size(); i++) {
    //printf("Inserting point %d\n", num++);
    if(points[i].idX != 0)
      printf("Node: Mass = %f, Center of Mass = (%f, %f, %f)\n", points[i].mass, points[i].pos.x, points[i].pos.y, points[i].pos.z);
      tree->insertNode(root, points[i], gridSize);
  };

  // print oct tree 
  //tree->printTree(root, 0);

  //tree->compute_center_of_mass(root);
  auto tree_build_time_end = chrono::steady_clock::now();
  profileStats->treeBuildTime += chrono::duration_cast<chrono::microseconds>(tree_build_time_end - tree_build_time_start);
  printf("Done Tree Build \n");
  auto iterative_step_time_start = chrono::steady_clock::now();

  auto tree_to_dfs_time_start = chrono::steady_clock::now();
  treeToDFSArray(root);
  printf("Number of dfsNodes %lu\n",dfsBHNodes.size());
  // order points in dfs order
  auto tree_to_dfs_time_end = chrono::steady_clock::now();
  profileStats->treeToDFSTime += chrono::duration_cast<chrono::microseconds>(tree_to_dfs_time_end - tree_to_dfs_time_start);
  std::cout << "Tree to DFS time: " << profileStats->treeToDFSTime.count() / 1000000.0 << " seconds." << std::endl;
  orderPointsDFS();
  printf("ordered points \n");

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  auto scene_build_time_start = chrono::steady_clock::now();

  auto auto_ropes_start = chrono::steady_clock::now();
  installAutoRopes(root);
  auto auto_ropes_end = chrono::steady_clock::now();
  profileStats->installAutoRopesTime += chrono::duration_cast<chrono::microseconds>(auto_ropes_end - auto_ropes_start);

  for(int i = 0; i < deviceBhNodes.size(); i++) {
    printf("deviceBhNodes[%d].mass = %f\n", i, deviceBhNodes[i].mass);
  }
  OWLBuffer deviceBhNodesBuffer = owlDeviceBufferCreate(context,OWL_USER_TYPE(deviceBhNodes[0]), deviceBhNodes.size(),deviceBhNodes.data());

  printf("Here has been hit\n");

  // create Triangles geomtery and create acceleration structure
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { /* sentinel to mark end of list */ }
  };  

  OWLGeomType trianglesGeomType = owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,module,"TriangleMesh");
  
  OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(vertices[0]), vertices.size(), vertices.data());
  OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(indices[0]), indices.size(), indices.data());
  OWLGeom trianglesGeom = owlGeomCreate(context,trianglesGeomType);
  
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,vertices.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,indices.size(),sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  OWLGroup world = owlInstanceGroupCreate(context,1,&trianglesGroup);
  owlGroupBuildAccel(world);

  auto scene_build_time_end = chrono::steady_clock::now();
  profileStats->sceneBuildTime += chrono::duration_cast<chrono::microseconds>(scene_build_time_end - scene_build_time_start);

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  auto intersections_setup_time_start = chrono::steady_clock::now();
  OWLVarDecl missProgVars[]= { { /* sentinel to mark end of list */ }};
  // ----------- create object  ----------------------------
  OWLMissProg missProg = owlMissProgCreate(context,module,"miss",sizeof(MissProgData), missProgVars,-1);

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
  
  OWLBuffer primaryLaunchRaysBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(orderedPrimaryLaunchRays[0]),
                            orderedPrimaryLaunchRays.size(),orderedPrimaryLaunchRays.data());

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

  // ##################################################################
  // Start Ray Tracing Parallel launch
  // ##################################################################
  auto start1 = std::chrono::steady_clock::now();
  owlLaunch2D(rayGen, NUM_POINTS, 1, lp);
  const float *rtComputedForces = (const float *)owlBufferGetPointer(ComputedForcesBuffer,0);
  auto end1 = std::chrono::steady_clock::now();
  profileStats->forceCalculationTime = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

  auto iterative_step_time_end = chrono::steady_clock::now();
  profileStats->iterativeStepTime += chrono::duration_cast<chrono::microseconds>(iterative_step_time_end - iterative_step_time_start);

  // ##################################################################
  // Output Force Computations
  // ##################################################################
  // compute real forces using cpu BH traversal
  printf("CPU OUTPUT!!!\n");
  printf("----------------------------------------\n");
  auto cpu_forces_start_time = chrono::steady_clock::now();
  //tree->computeForces(root, points, cpuComputedForces);
  auto cpu_forces_end_time = chrono::steady_clock::now();
  profileStats->cpuForceCalculationTime += chrono::duration_cast<chrono::microseconds>(cpu_forces_end_time - cpu_forces_start_time);
  printf("----------------------------------------\n");

  int pointsFailing = 0;
  // for(int i = 0; i < NUM_POINTS; i++) {
  //   float percent_error = (abs((rtComputedForces[i] - cpuComputedForces[i])) / cpuComputedForces[i]) * 100.0f;
  //   if(percent_error > 5.0f) {
  //     // LOG_OK("++++++++++++++++++++++++");
  //     // LOG_OK("POINT #" << i << ", (" << points[i].x << ", " << points[i].y << ") , HAS ERROR OF " << percent_error << "%");
  //     // LOG_OK("++++++++++++++++++++++++");
  //     // printf("RT force = %f\n", rtComputedForces[i]);
  //     // printf("CPU force = %f\n", cpuComputedForces[i]);
  //     // LOG_OK("++++++++++++++++++++++++");
  //     // printf("\n");
  //     pointsFailing++;
  //   }
  // }
  printf("Points failing percent error: %d\n", pointsFailing);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  auto total_run_time_end = chrono::steady_clock::now();
  profileStats->totalProgramTime += chrono::duration_cast<chrono::microseconds>(total_run_time_end - total_run_time);

  // free memory

  // Print Statistics
  printf("--------------------------------------------------------------\n");
  std::cout << "Tree build time: " << profileStats->treeBuildTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Tree to DFS time: " << profileStats->treeToDFSTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Install AutoRopes time: " << profileStats->installAutoRopesTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Intersections setup time: " << profileStats->intersectionsSetupTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "RT Cores Force Calculations time: " << profileStats->forceCalculationTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "CPU Force Calculations time: " << profileStats->cpuForceCalculationTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Iterative Step time: " << profileStats->iterativeStepTime.count() / 1000000.0 << " seconds." << std::endl;
  std::cout << "Total Program time: " << profileStats->totalProgramTime.count() / 1000000.0 << " seconds." << std::endl;
  printf("--------------------------------------------------------------\n");

  // destory owl stuff
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
}