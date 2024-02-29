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
float totalMemoryNeeded = 0.0f;

int gpuDeviceID = 1;
OWLContext context = owlContextCreate(&gpuDeviceID, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

void treeToDFSArray(Node* root) {
  std::stack<Node*, std::vector<Node*>> nodeStack;
  int dfsIndex = 0;
  float triangleXLocation = 0.0f;
  float level = 0.0f;
  int currentIndex = 0;
  int primIDIndex = 1;
  vertices.reserve((NUM_POINTS * 2) * 3);
  indices.reserve(NUM_POINTS * 2);
  deviceBhNodes.reserve(NUM_POINTS * 2);
  dfsBHNodes.reserve(NUM_POINTS * 2);

  nodeStack.push(root);
  while (!nodeStack.empty()) {
    Node* currentNode = nodeStack.top();
    nodeStack.pop();
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
    currentBhNode.centerOfMassX = currentNode->cofm.x;
    currentBhNode.centerOfMassY = currentNode->cofm.y;
    currentBhNode.centerOfMassZ = currentNode->cofm.z;
    currentBhNode.isLeaf = (currentNode->type == bhLeafNode) ? 1 : 0;
    currentBhNode.nextRayLocation_x = triangleXLocation;
    currentBhNode.nextRayLocation_y = level;
    currentBhNode.nextPrimId = primIDIndex;
    currentBhNode.numParticles = currentNode->particles.size();
    for(int i = 0; i < currentNode->particles.size(); i++) {
      currentBhNode.particles[i] = currentNode->particles[i];
    }

    deviceBhNodes.push_back(currentBhNode);
    primIDIndex++;

    // add bhNode to DFS array
    currentNode->dfsIndex = dfsIndex;
    dfsBHNodes.push_back(currentNode);
    dfsIndex++;

    // Enqueue child nodes in the desired order
    for(int i = 7; i >= 0; i--) {
      if(currentNode->children[i] != nullptr) {
        nodeStack.push(currentNode->children[i]);
      }
    }
  }
}

void orderPointsDFS() {
  int primaryRayIndex = 0;
  for(int i = 0; i < dfsBHNodes.size(); i++) {
    if(dfsBHNodes[i]->type == bhLeafNode) {
      for(int j = 0; j < dfsBHNodes[i]->particles.size(); j++) {
        orderedPrimaryLaunchRays[primaryRayIndex].pointID = dfsBHNodes[i]->particles[j];
        orderedPrimaryLaunchRays[primaryRayIndex].primID = 0;
        orderedPrimaryLaunchRays[primaryRayIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
        primaryRayIndex++;
      }
    }
  }
}

int maxDepth(Node* node) {
    if (node == nullptr) {
        return 0;
    } else {
        int maxChildDepth = 0;
        for (Node* child : node->children) {
            int childDepth = maxDepth(child);
            if (childDepth > maxChildDepth) {
                maxChildDepth = childDepth;
            }
        }
        return 1 + maxChildDepth;
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
          deviceBhNodes[dfsIndex].autoRopeRayLocation_x = deviceBhNodes[ropeIndex-1].nextRayLocation_x;
          deviceBhNodes[dfsIndex].autoRopeRayLocation_y = deviceBhNodes[ropeIndex-1].nextRayLocation_y;
          deviceBhNodes[dfsIndex].autoRopePrimId = deviceBhNodes[ropeIndex-1].nextPrimId;
        } else {
          deviceBhNodes[dfsIndex].autoRopeRayLocation_x = 0.0f;
          deviceBhNodes[dfsIndex].autoRopeRayLocation_y = 0.0f;
          deviceBhNodes[dfsIndex].autoRopePrimId = 0;
        }
      } else {
        initial = 1;
      } 

      // Enqueue child nodes in the desired order
      for(int i = 7; i >= 0; i--) {
        if(currentNode->children[i] != nullptr) {
          ropeStack.push(currentNode->children[i]);
        }
      }
      iterations++;
    }

    printf("Max stack size: %d\n", maxStackSize);
    printf("Iterations: %d\n", iterations);
}

int main(int ac, char **av) {
  LOG("Starting");

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
        //p.vel.x = 0.0f;
        //p.vel.y= 0.0f;
        //p.vel.z = 0.0f;
        p.mass = disMass(gen);
        p.idX = numPointsSoFar;

        fprintf(outFile, "%f %f %f %f %f %f %f\n", p.mass, p.pos.x, p.pos.y, p.pos.z, nullptr, nullptr, nullptr);
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
          //point.vel.x = velRead.x;
          //point.vel.y = velRead.u;
          //point.vel.z = velRead.z;
          point.mass = mass;
          //printf("Point # %d has x = %f, y = %f, mass = %f\n", launchIndex, point.pos.x, point.pos.y, point.mass);
          point.idX = launchIndex;

          points.push_back(point);
          primaryLaunchRays[launchIndex].pointID = launchIndex;
          primaryLaunchRays[launchIndex].primID = 0;
          primaryLaunchRays[launchIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
          launchIndex++;
      }
      fclose(inFile);
    } else if(std::string(av[1]) == "csv") {
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
      std::ifstream file(av[2]);
      if (!file.is_open()) {
          std::cerr << "Error opening file!" << std::endl;
          return 1;
      }

      std::string line;
      int launchIndex = 0;
      while(std::getline(file, line)) {
        //printf("Line: %s\n", line.c_str());
        // split line by comma
        std::istringstream ss(line);
        std::vector<std::string> tokens;
        std::string token;
        int index = 0;
        Point point;
        while (std::getline(ss, token, ',')) {
            switch (index) {
                case 0: point.pos.x = std::stof(token) * 1000.0f; break; 
                case 1: point.pos.y = std::stof(token) * 1000.0f; break;
                case 2: point.pos.z = std::stof(token) * 1000.0f; break;
                case 3: point.mass = std::stof(token) * 10000.0f; break;
                default: printf("Error reading csv file\n"); break;
            }
            tokens.push_back(token);
            //printf("Token: %s\n", tokens[index].c_str());
            index++;
        }

        point.idX = launchIndex;
        fprintf(outFile, "%f %f %f %f %f %f %f\n", point.mass, point.pos.x, point.pos.y, point.pos.z, nullptr, nullptr, nullptr);
        points.push_back(point);
        primaryLaunchRays[launchIndex].pointID = launchIndex;
        primaryLaunchRays[launchIndex].primID = 0;
        primaryLaunchRays[launchIndex].orgin = vec3f(0.0f, 0.0f, 0.0f);
        launchIndex++;
        if(launchIndex == NUM_POINTS) {
          break;
        }
      }
      printf("Number of points: %lu\n", points.size());
      fclose(outFile);
    } 
    else {
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
  Node* root = new Node(0.0f, 0.0f, 0.0f, gridSize, -1);
  root->type = bhNonLeafNode;

  LOG("Bulding Non Bucket Tree with # Of Bodies = " << points.size());
  auto tree_build_time_start = chrono::steady_clock::now();
  for(int i = 0; i < points.size(); i++) {
    Node* new_node = new Node(points[i].pos.x, points[i].pos.y, points[i].pos.z, gridSize * 0.5, points[i].idX);
    new_node->mass = points[i].mass;
    tree->insertNode(root, new_node, gridSize * 0.5);
  };

  printf("Max depth of Non Bucket Tree: %d\n", maxDepth(root));

  printf("Done building!\n");

  std::vector<Node*> leaves;
  float minS = GRID_SIZE + 1.0f;
  tree->traverseOctreeDFS(root, leaves, &minS);
  printf("Min S: %f\n", minS);

  // create new tree with bucket sized leaf nodes
  root = new Node(0.0f, 0.0f, 0.0f, gridSize, -1);
  root->type = bhNonLeafNode;

  int numLeaves;
  numLeaves = std::ceil(leaves.size() / double(BUCKET_SIZE));
  printf("Number of leaves: %d\n", numLeaves);

  for(int i = 0; i < numLeaves; i++) {
    Node* new_node = new Node(0.0f, 0.0f, 0.0f, gridSize * 0.5, -1);
    vec3f cofm(0.0f, 0.0f, 0.0f);
    float mass = 0.0f;
    for(int j = 0; j < BUCKET_SIZE; j++) {
      if((i * BUCKET_SIZE) + j < leaves.size()) {
        cofm.x += leaves[(i * BUCKET_SIZE) + j]->cofm.x * leaves[(i * BUCKET_SIZE) + j]->mass;
        cofm.y += leaves[(i * BUCKET_SIZE) + j]->cofm.y * leaves[(i * BUCKET_SIZE) + j]->mass;
        cofm.z += leaves[(i * BUCKET_SIZE) + j]->cofm.z * leaves[(i * BUCKET_SIZE) + j]->mass;
        mass += leaves[(i * BUCKET_SIZE) + j]->mass;
        new_node->particles.push_back(leaves[(i * BUCKET_SIZE) + j]->pointID);
      }
    }
    cofm.x /= mass;
    cofm.y /= mass;
    cofm.z /= mass;
    new_node->cofm.x = cofm.x;
    new_node->cofm.y = cofm.y;
    new_node->cofm.z = cofm.z;
    new_node->mass = mass;
    tree->insertNode(root, new_node, gridSize * 0.5);
  }
  printf("Max depth of Bucket Tree: %d\n", maxDepth(root));
  tree->computeCOM(root);
  ;
  auto tree_build_time_end = chrono::steady_clock::now();
  profileStats->treeBuildTime += chrono::duration_cast<chrono::microseconds>(tree_build_time_end - tree_build_time_start);
  auto iterative_step_time_start = chrono::steady_clock::now();

  auto tree_to_dfs_time_start = chrono::steady_clock::now();
  treeToDFSArray(root);
  printf("Number of dfsNodes %lu\n",dfsBHNodes.size());
  // order points in dfs order
  auto tree_to_dfs_time_end = chrono::steady_clock::now();
  profileStats->treeToDFSTime += chrono::duration_cast<chrono::microseconds>(tree_to_dfs_time_end - tree_to_dfs_time_start);
  std::cout << "Tree to DFS time: " << profileStats->treeToDFSTime.count() / 1000000.0 << " seconds." << std::endl;
  orderPointsDFS();

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  auto scene_build_time_start = chrono::steady_clock::now();

  auto auto_ropes_start = chrono::steady_clock::now();
  installAutoRopes(root);
  auto auto_ropes_end = chrono::steady_clock::now();
  profileStats->installAutoRopesTime += chrono::duration_cast<chrono::microseconds>(auto_ropes_end - auto_ropes_start);


  // create Triangles geomtery and create acceleration structure
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { /* sentinel to mark end of list */ }
  };  

  OWLGeomType trianglesGeomType = owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,module,"TriangleMesh");
  
  //printf("Size of malloc for vertices is %.2f gb.\n", (vertices.size() * sizeof(vec3f))/1000000000.0);
  totalMemoryNeeded += (vertices.size() * sizeof(vec3f)) / 1000000000.0;
  OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(vertices[0]), vertices.size(), vertices.data());
  //printf("Size of malloc for indices is %.2f gb.\n", (indices.size() * sizeof(vec3i))/1000000000.0);
  totalMemoryNeeded += (indices.size() * sizeof(vec3i)) / 1000000000.0;
  OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(indices[0]), indices.size(), indices.data());
  OWLGeom trianglesGeom = owlGeomCreate(context,trianglesGeomType);
  
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,vertices.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,indices.size(),sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  size_t accelSize;
  owlGroupGetAccelSize(trianglesGroup, &accelSize, nullptr);
  printf("Accel size: %f gb\n", accelSize/1000000000.0f);
  totalMemoryNeeded += accelSize/1000000000.0f;
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

  //printf("Size of malloc for points is %.2f gb.\n", (points.size() * sizeof(Point))/1000000000.0);
  totalMemoryNeeded += (points.size() * sizeof(Point))/1000000000.0;
  OWLBuffer DevicePointsBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());
  owlParamsSetBuffer(lp, "devicePoints", DevicePointsBuffer);

  //printf("Size of malloc for bhNodes is %.2f gb.\n", (deviceBhNodes.size() * sizeof(deviceBhNode))/1000000000.0);
  totalMemoryNeeded += (deviceBhNodes.size() * sizeof(deviceBhNode))/1000000000.0;
  OWLBuffer DeviceBhNodesBuffer = owlDeviceBufferCreate( 
     context, OWL_USER_TYPE(deviceBhNodes[0]), deviceBhNodes.size(), deviceBhNodes.data());
  owlParamsSetBuffer(lp, "deviceBhNodes", DeviceBhNodesBuffer);

  //printf("Size of malloc for computedForces is %.2f gb.\n", (computedForces.size() * sizeof(float))/1000000000.0);
  totalMemoryNeeded += (computedForces.size() * sizeof(float))/1000000000.0;
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
  
  //printf("Size of malloc for orderedPrimaryLaunchRays is %.2f gb.\n", (orderedPrimaryLaunchRays.size() * sizeof(CustomRay))/1000000000.0);
  totalMemoryNeeded += (orderedPrimaryLaunchRays.size() * sizeof(CustomRay))/1000000000.0;
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

  printf("Minimum node-s value %f\n", minNodeSValue);
  printf("Total memory needed: %.2f gb\n", totalMemoryNeeded);
  
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
  for(int i = 0; i < NUM_POINTS; i++) {
    float percent_error = (abs((rtComputedForces[i] - cpuComputedForces[i])) / cpuComputedForces[i]) * 100.0f;
    if(percent_error > 5.0f) {
      // LOG_OK("++++++++++++++++++++++++");
      // LOG_OK("POINT #" << i << ", (" << points[i].pos.x << ", " << points[i].pos.y << ") , HAS ERROR OF " << percent_error << "%");
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