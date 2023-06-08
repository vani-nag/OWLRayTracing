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
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];
const vec3f lookFrom(13, 2, 3);
const vec3f lookAt(0, 0, 0);
const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

// random init
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(-100.0f, 100.0f);  // Range for X and Y coordinates
std::uniform_real_distribution<float> disMass(0.1f, 20.0f);  // Range mass

// global variables
int deviceID;
vec2f fbSize;
float gridSize = 200.0f;
float threshold = 0.5f;

OWLContext context = owlContextCreate(nullptr, 1);
OWLModule module = owlModuleCreate(context, deviceCode_ptx);

OWLVarDecl SpheresGeomVars[] = {
    {"prims", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, prims)},
    {"rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom, rad)},
    {/* sentinel to mark end of list */}};

OWLGeomType SpheresGeomType = owlGeomTypeCreate(
    context, OWL_GEOMETRY_USER, sizeof(SpheresGeom), SpheresGeomVars, -1);

// OWLVarDecl PointIntersectionInfoVars[] = {
//     {"body", OWL_BUFPTR, OWL_OFFSETOF(PointIntersectionInfo, body)},
//     {"didIntersectNodes", OWL_BUFPTR, OWL_OFFSETOF(PointIntersectionInfo, didIntersectNodes)},
//     {"bhNodes", OWL_BUFPTR, OWL_OFFSETOF(PointIntersectionInfo, bhNodes)},
//     {/* sentinel to mark end of list */}};

// OWLGeomType PointIntersectionInfoType = owlGeomTypeCreate(
//     context, OWL_GEOMETRY_USER, sizeof(PointIntersectionInfo), PointIntersectionInfoVars, -1);

// ##################################################################
// Create world function
// ##################################################################
OptixTraversableHandle createSceneGivenGeometries(std::vector<Sphere> Spheres, float spheresRadius) {
  // Init Frame Buffer. Don't need 2D threads, so just use x-dim for threadId
  fbSize = vec2f(Spheres.size(), 1);

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

int main(int ac, char **av) {

  auto total_run_time = std::chrono::steady_clock::now();

  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################
  BarnesHutTree* tree = new BarnesHutTree(threshold, gridSize);
  Node* root = new Node(0.f, 0.f, gridSize);

  // Generate random points
  int numPoints = 2;  // Specify the number of points

  std::vector<Point> points;

  for (int i = 0; i < numPoints; ++i) {
    Point p;
    p.x = dis(gen);
    p.y = dis(gen);
    p.mass = disMass(gen);
    p.idX = i;
    points.push_back(p);
    printf("Point # %d has x = %f, y = %f, mass = %f\n", i, p.x, p.y, p.mass);
  }

  OWLBuffer PointsBuffer = owlDeviceBufferCreate(
     context, OWL_USER_TYPE(points[0]), points.size(), points.data());


  LOG("Bulding Tree with # Of Bodies = " << points.size());
  auto start_tree = std::chrono::steady_clock::now();
  for(const auto& point: points) {
    tree->insertNode(root, point);
  };

  auto end_tree = std::chrono::steady_clock::now();
  auto elapsed_tree = std::chrono::duration_cast<std::chrono::microseconds>(end_tree - start_tree);
  std::cout << "Barnes Hut Tree Build Time: " << elapsed_tree.count() / 1000000.0 << " seconds."
            << std::endl;

  //LOG("Size of tree = " << calculateQuadTreeSize(root));
  tree->printTree(root, 0);

  // Get the device ID
  cudaGetDevice(&deviceID);
  LOG_OK("Device ID: " << deviceID);
  owlGeomTypeSetIntersectProg(SpheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(SpheresGeomType, module, "Spheres");
  owlBuildPrograms(context);

  OWLBuffer frameBuffer = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x);

  OWLVarDecl myGlobalsVars[] = {
    {"yIDx", OWL_INT, OWL_OFFSETOF(MyGlobals, yIDx)},
    {"parallelLaunch", OWL_INT, OWL_OFFSETOF(MyGlobals, parallelLaunch)},
    {"levelIntersectionData", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, levelIntersectionData)},
    {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);

  // ##################################################################
  // Level order traversal of Barnes Hut Tree to build worlds
  // ##################################################################

  std::vector<LevelIntersectionInfo> levelIntersectionData;
  std::vector<OptixTraversableHandle> worlds;
  std::vector<Sphere> InternalSpheres;
  std::vector<Node> InternalNodes;
  LevelIntersectionInfo levelInfo;
  float prevS = gridSize;
  int level = 0;

  // level order traversal of BarnesHutTree
  // Create an empty queue for level order traversal
  queue<Node*> q;
  
  LOG("Bulding OWL Scenes");
  // Enqueue Root and initialize height
  q.push(root);
  auto start_b = std::chrono::steady_clock::now();
  while (q.empty() == false) {
      // Print front of queue and remove it from queue
      Node* node = q.front();
      if((node->s != prevS)) {
        if(!InternalSpheres.empty()) {
          worlds.push_back(createSceneGivenGeometries(InternalSpheres, (gridSize / threshold)));
          for(int i = 0; i < points.size(); i++) {
            levelInfo.pointIntersectionInfo[i].body = points[i];
            //levelInfo.pointIntersectionInfo[level-1].didIntersectNodes = boolArray;
            std::copy(InternalNodes.begin(), InternalNodes.end(), levelInfo.pointIntersectionInfo[i].bhNodes);
          }
          levelInfo.level = level + 1;
          levelIntersectionData.push_back(levelInfo);
          // Point body = levelIntersectionData[level].pointIntersectionInfo[0].body;
          // printf("Level is %d, Body x is %f and y is %f \n", level+1, body.x, body.y);
          // body = levelIntersectionData[level].pointIntersectionInfo[1].body;
          // printf("Level is %d, Body x is %f and y is %f \n", level+1, body.x, body.y);
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
          InternalSpheres.push_back(Sphere{vec3f{node->centerOfMassX, node->centerOfMassY, 0}, node->mass, false});
          InternalNodes.push_back((*node));
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

  if(!InternalSpheres.empty()) {
    worlds.push_back(createSceneGivenGeometries(InternalSpheres, (gridSize / threshold)));
    for(int i = 0; i < points.size(); i++) {
      levelInfo.pointIntersectionInfo[i].body = points[i];
      //levelInfo.pointIntersectionInfo[level-1].didIntersectNodes = boolArray;
      std::copy(InternalNodes.begin(), InternalNodes.end(), levelInfo.pointIntersectionInfo[i].bhNodes);
    }
    levelInfo.level = level+1;
    levelIntersectionData.push_back(levelInfo);
    // Point body = levelIntersectionData[level].pointIntersectionInfo[0].body;
    // printf("Level is %d, Body x is %f and y is %f \n", level+1, body.x, body.y);
    // body = levelIntersectionData[level].pointIntersectionInfo[1].body;
    // printf("Level is %d, Body x is %f and y is %f \n", level+1, body.x, body.y);
  }

  // printf("++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  // for(int z = 0; z < level+1; z++) {
  //   Point body = levelIntersectionData.data()[z].pointIntersectionInfo[0].body;
  //   printf("Level is %d, Body x is %f and y is %f \n", z+1, body.x, body.y);
  //   body = levelIntersectionData.data()[z].pointIntersectionInfo[1].body;
  //   printf("Level is %d, Body x is %f and y is %f \n", z+1, body.x, body.y);
  // }

  OWLBuffer IntersectionsBuffer = owlManagedMemoryBufferCreate( 
     context, OWL_USER_TYPE(levelIntersectionData[0]), levelIntersectionData.size(), levelIntersectionData.data());
  
  // for(int i = 0; i < worlds.size(); i++) {
  //   LOG_OK("Level #: " << i + 1);
  //   for(int k = 0; k < points.size(); k++) {
  //     LOG("Point #: " << k + 1);
  //     for(int j = 0; j < levelIntersectionData[i]->pointIntersectionInfo[k].size(); j++) {
  //       std::cout << "didIntersect is: " << std::boolalpha << pointsIntersectionData[i]->didIntersectNodes[k][j] << std::endl;
  //       Node *tempNode = pointsIntersectionData[i]->bhNodes[k][j];
  //       std::cout << "Node: Mass = " << tempNode->mass << ", Center of Mass = (" << tempNode->centerOfMassX << ", " << tempNode->centerOfMassY << ")\n";
  //     }
  //   }
  // }

  auto end_b = std::chrono::steady_clock::now();
  auto elapsed_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
  std::cout << "OWL Scenes Build time: " << elapsed_b.count() / 1000000.0 << " seconds."
            << std::endl;
  
  std::cout << "Worlds size:" << worlds.size() << std::endl;
  OWLBuffer WorldsBuffer = owlDeviceBufferCreate(
        context, OWL_USER_TYPE(worlds[0]), worlds.size(), worlds.data());

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
      //{"internalSpheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, internalSpheres)},
      {"points", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, points)},
      {"fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize)},
      {"worlds", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, worlds)},
      {/* sentinel to mark end of list */}};

  // ........... create object  ............................
  OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                                     sizeof(RayGenData), rayGenVars, -1);\
  
                                     

  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen, "points", PointsBuffer);
  owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
  owlRayGenSetBuffer(rayGen, "worlds", WorldsBuffer);

  // programs have been built before, but have to rebuild raygen and
  // miss progs

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // Start Ray Tracing Parallel launch
  // ##################################################################
  // auto start1 = std::chrono::steady_clock::now();
  // owlParamsSet1i(lp, "parallelLaunch", 1);
  // owlLaunch2D(rayGen, points.size(), worlds.size(), lp);

  // auto end1 = std::chrono::steady_clock::now();
  // auto elapsed1 =
  //     std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  // std::cout << "Intersections time for parallel launch: " << elapsed1.count() / 1000000.0
  //           << " seconds." << std::endl;
  
  // ##################################################################
  // Start Ray Tracing Series launch
  // ##################################################################
  auto start2 = std::chrono::steady_clock::now();
  owlParamsSet1i(lp, "parallelLaunch", 0);
  owlParamsSetBuffer(lp, "levelIntersectionData", IntersectionsBuffer);
  for(int l = 1; l < 2; l++) {
    owlParamsSet1i(lp, "yIDx", l);
    owlLaunch2D(rayGen, points.size(), 1, lp);

    // calculate forces here
    const LevelIntersectionInfo *intersectionsOutputData = (const LevelIntersectionInfo*)owlBufferGetPointer(IntersectionsBuffer,0);
    printf("==========================================\n");
    for(int i = 0; i < 2; i++) {
      printf("++++++++++++++++++++++++++++++++++++++++\n");
      printf("Point # %d with x = %f, y = %f, mass = %f\n", i, points[i].x, points[i].y, points[i].mass);
      for(int k = 0; k < 4; k++) {
        if(intersectionsOutputData[l].pointIntersectionInfo[i].didIntersectNodes[k] != 0) {
          Node bhNode = intersectionsOutputData[l].pointIntersectionInfo[i].bhNodes[k];
          printf("Intersected bhNode with x = %f, y = %f, mass = %f\n", bhNode.centerOfMassX, bhNode.centerOfMassY, bhNode.mass);
        }
      }
    }
    printf("==========================================\n");
  }

  auto end2 = std::chrono::steady_clock::now();
  auto elapsed2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
  std::cout << "Intersections time for series launch: " << elapsed2.count() / 1000000.0
            << " seconds." << std::endl;
  
  // const uint32_t *fb
  //   = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  auto total_run_time_end = std::chrono::steady_clock::now();
  auto elapsed_run_time_end = std::chrono::duration_cast<std::chrono::microseconds>(total_run_time_end - total_run_time);
  std::cout << "Total run time is: " << elapsed_run_time_end.count() / 1000000.0 << " seconds."
            << std::endl;
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
}