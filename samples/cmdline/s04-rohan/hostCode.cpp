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

std::vector<Sphere> InternalSpheres;
std::vector<Sphere> LeafSpheres;

int main(int ac, char **av) {

  // ##################################################################
  // Building Barnes Hut Tree
  // ##################################################################

  BarnesHutTree* tree = new BarnesHutTree(0.5f, 5.0f);
  Node* root = new Node(0.f, 0.f, 5.0f);

  std::vector<Point> points = {
    {3.0, 4.0, 10},
    {4.0, 1.0, 6.0},
    {3.0, -3.0, 4.0},
    {1.0, -1.0, 1.0},
    {-2.0, 4.0, 8.0}
  };
  
  for(const auto& point: points) {
    tree->insertNode(root, point);
  };

  tree->printTree(root, 0);

  // ##################################################################
  // Create scene
  // ##################################################################

  float sphereRadius = 2.5/.5;

  for(const auto& point: points) {
    LeafSpheres.push_back(Sphere{vec3f{point.x, point.y, 0}, point.mass, true});
  }

  // level order traversal of BarnesHutTree
  // Create an empty queue for level order traversal
  queue<Node*> q;

  // Enqueue Root and initialize height
  q.push(root);

  while (q.empty() == false) {
      // Print front of queue and remove it from queue
      Node* node = q.front();
      if(node->s == 2.5f) {
        if(node->mass != 0.0f) {
          InternalSpheres.push_back(Sphere{vec3f{node->centerOfMassX, node->centerOfMassY, 0}, node->mass, false});
        }
      }
      //std::cout << node->s << " ";
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

  // Init Frame Buffer. Don't need 2D threads, so just use x-dim for threadId
  const vec2i fbSize(InternalSpheres.size(), 1);

  LOG_OK(" Executing DBSCAN");
  LOG_OK(" dataset size: " << InternalSpheres.size());

  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate(nullptr, 1);
  OWLModule module = owlModuleCreate(context, deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  OWLVarDecl SpheresGeomVars[] = {
      {"prims", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, prims)},
      {"rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom, rad)},
      {/* sentinel to mark end of list */}};

  OWLGeomType SpheresGeomType = owlGeomTypeCreate(
      context, OWL_GEOMETRY_USER, sizeof(SpheresGeom), SpheresGeomVars, -1);
  /*owlGeomTypeSetClosestHit(SpheresGeomType,0,
                           module,"Spheres");*/
  owlGeomTypeSetIntersectProg(SpheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(SpheresGeomType, module, "Spheres");

  // make sure to do that *before* setting up the geometry, since the
  // user geometry group will need the compiled bounds programs upon
  // accelBuild()
  owlBuildPrograms(context);
  LOG_OK("BUILD prog DONE\n");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  // LOG("building geometries ...");

  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x);

  
  OWLBuffer LeafSpheresBuffer = owlDeviceBufferCreate(
      context, OWL_USER_TYPE(LeafSpheres[0]), LeafSpheres.size(), LeafSpheres.data());

  OWLGeom LeafSpheresGeom = owlGeomCreate(context, SpheresGeomType);
  owlGeomSetPrimCount(LeafSpheresGeom, LeafSpheres.size());
  owlGeomSetBuffer(LeafSpheresGeom, "prims", LeafSpheresBuffer);
  owlGeomSet1f(LeafSpheresGeom, "rad", sphereRadius);

  OWLBuffer InternalSpheresBuffer = owlDeviceBufferCreate(
      context, OWL_USER_TYPE(InternalSpheres[0]), InternalSpheres.size(), InternalSpheres.data());

  OWLGeom InternalSpheresGeom = owlGeomCreate(context, SpheresGeomType);
  owlGeomSetPrimCount(InternalSpheresGeom, InternalSpheres.size());
  owlGeomSetBuffer(InternalSpheresGeom, "prims", InternalSpheresBuffer);
  owlGeomSet1f(InternalSpheresGeom, "rad", sphereRadius);

  // ##################################################################
  // Params
  // ##################################################################

  OWLVarDecl myGlobalsVars[] = {
      // {"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, frameBuffer)},
      // {"callNum", OWL_INT, OWL_OFFSETOF(MyGlobals, callNum)},
      // {"minPts", OWL_INT, OWL_OFFSETOF(MyGlobals, minPts)},
      {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);

  LOG_OK("Geoms and Params DONE\n");

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
 
  OWLGeom userGeoms[] = {InternalSpheresGeom, LeafSpheresGeom};

  auto start_b = std::chrono::steady_clock::now();
  OWLGroup spheresGroup = owlUserGeomGroupCreate(context, 1, userGeoms);
  owlGroupBuildAccel(spheresGroup);

  OWLGroup world = owlInstanceGroupCreate(context, 1, &spheresGroup);
  owlGroupBuildAccel(world);

  LOG_OK("Group build DONE\n");

  auto end_b = std::chrono::steady_clock::now();
  auto elapsed_b =
      std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
  std::cout << "Build time: " << elapsed_b.count() / 1000000.0 << " seconds."
            << std::endl;

  // ##################################################################
  // set miss and raygen programs
  // ##################################################################


  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
      {"internalSpheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, internalSpheres)},
      {"leafSpheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, leafSpheres)},
      {"fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize)},
      {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
      {"camera.org", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.origin)},
      {"camera.llc", OWL_FLOAT3,
       OWL_OFFSETOF(RayGenData, camera.lower_left_corner)},
      {"camera.horiz", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.horizontal)},
      {"camera.vert", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.vertical)},
      {/* sentinel to mark end of list */}};

  // ........... create object  ............................
  OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                                     sizeof(RayGenData), rayGenVars, -1);

  // ........... compute variable values  ..................
  const float vfov = fovy;
  const vec3f vup = lookUp;
  const float aspect = fbSize.x / float(fbSize.y);
  const float theta = vfov * ((float)M_PI) / 180.0f;
  const float half_height = tanf(theta / 2.0f);
  const float half_width = aspect * half_height;
  const float focusDist = 10.f;
  const vec3f origin = lookFrom;
  const vec3f w = normalize(lookFrom - lookAt);
  const vec3f u = normalize(cross(vup, w));
  const vec3f v = cross(w, u);
  const vec3f lower_left_corner = origin - half_width * focusDist * u -
                                  half_height * focusDist * v - focusDist * w;
  const vec3f horizontal = 2.0f * half_width * focusDist * u;
  const vec3f vertical = 2.0f * half_height * focusDist * v;

  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen, "internalSpheres", InternalSpheresBuffer);
  owlRayGenSetBuffer(rayGen, "leafSpheres", LeafSpheresBuffer);
  owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
  owlRayGenSetGroup(rayGen, "world", world);
  owlRayGenSet3f(rayGen, "camera.org", (const owl3f &)origin);
  owlRayGenSet3f(rayGen, "camera.llc", (const owl3f &)lower_left_corner);
  owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f &)horizontal);
  owlRayGenSet3f(rayGen, "camera.vert", (const owl3f &)vertical);

  // programs have been built before, but have to rebuild raygen and
  // miss progs

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // Start Ray Tracing
  // ##################################################################

  auto start1 = std::chrono::steady_clock::now();

  owlLaunch2D(rayGen, LeafSpheres.size(), 1, lp);

  auto end1 = std::chrono::steady_clock::now();
  auto elapsed1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cout << "Core points time: " << elapsed1.count() / 1000000.0
            << " seconds." << std::endl;
  
  const uint32_t *fb
    = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);

  //Write cluster results to file
  // ofile << "ind" << '\t'<< "cluster"<< std::endl;
  /*for(int i = 0; i < Spheres.size(); i++)
  {
          temp = find(fb[i].parent,fb);

                  ofile << i << '\t'<< temp<< std::endl;
          //cout<<i<<'\t'<<find(fb[i].parent,fb)<<'\n';
  }*/

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);

  
}