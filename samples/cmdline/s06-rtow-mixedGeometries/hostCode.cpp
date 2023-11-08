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

// This program renders the recursive Sierpinski tetrahedron to a given depth.
// The code demonstrates how to create nested instances.

// public owl API
#include "GeomTypes.h"        //here
#include <owl/DeviceMemory.h> //here
#include <owl/owl.h>

// our device-side data structures
// #include "deviceCode.h"      //here
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <vector>

// here
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>

#define FLOAT_MAX 3.402823466e+38

#define LOG(message)                                                           \
  std::cout << OWL_TERMINAL_BLUE;                                              \
  std::cout << "#owl.sample(main): " << message << std::endl;                  \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                                        \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                                        \
  std::cout << "#owl.sample(main): " << message << std::endl;                  \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "s08-sierpinski.png";
// const vec2i fbSize(800,600);
const vec3f lookFrom(13, 2, 3);
const vec3f lookAt(0, 0, 0);
const vec3f lookUp(0.f, 1.f, 0.f);
const float fovy = 20.f;

std::vector<Sphere> Spheres;
std::vector<Sphere> updatedSpheres;
std::vector<Sphere> tempSpheres;
std::vector<Neigh> neighbors;

/*Command-line args
argv[1] = input file
argv[2] = number of points to read from file
argv[3] = dataset dimension
argv[4] = start radius
argv[5] = 'k' value for knn
argv[6] = file to write execution time
*/

int main(int ac, char **argv) {

  // Read from input .csv file
	/* Input must be in the format:
	(x1,y1,z1)
	(x2,y2,z2)... 
	*/

  std::string line;
  std::ifstream myfile;
  myfile.open(argv[1]);
  if (!myfile.is_open()) {
    perror("Error open");
    exit(EXIT_FAILURE);
  }
  std::vector<float> vect;
	int dim = atoi(argv[3]);

	// Read x,y,z co-ordinates of `count` lines
  int count = atoi(argv[2]) * dim;
  while (getline(myfile, line) && count > 0) {
    std::stringstream ss(line);
    float i;
    while (ss >> i) {
      vect.push_back(i);
      count--;
      if (ss.peek() == ',')
        ss.ignore();
    }
  }
  // ##################################################################
  // Create scene
  // ##################################################################

  // Select minPts,k
  float radius = atof(argv[4]); 
  int knn = atoi(argv[5]);
  // vec3f org = vec3f(vect.at(0), vect.at(1), vect.at(2));

	//If dataset is 2D, set z dimension to 0 when creating spheres
	if (dim == 2) {
		for (int i = 0, j = 0; i < vect.size(); i += 2, j += 1)
    	Spheres.push_back(Sphere{vec3f(vect.at(i), vect.at(i + 1), 0)});
	}

	if (dim == 3) {
		for (int i = 0, j = 0; i < vect.size(); i += 3, j += 1)
    	Spheres.push_back(Sphere{vec3f(vect.at(i), vect.at(i + 1), vect.at(i + 2))});
	}
  

  // Init neighbors array that maintains list of k nearest neighbors
  for (int j = 0; j < Spheres.size(); j++) {
    for (int i = 0; i < knn; i++)
      neighbors.push_back(Neigh{-1, FLOAT_MAX, knn});
  }

  // Frame Buffer -- x coordinate is number of paralell rays generated
  vec2i fbSize(Spheres.size(), 1);

  LOG_OK(" num spheres: " << Spheres.size());

  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate(nullptr, 1);
  OWLModule module = owlModuleCreate(context, ptxCode);
  LOG_OK("init DONE\n");

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  OWLVarDecl SpheresGeomVars[] = {
      {"prims", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, prims)},
      {"rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom, rad)},
      {/* sentinel to mark end of list */}};

  OWLGeomType SpheresGeomType = owlGeomTypeCreate(
      context, OWL_GEOMETRY_USER, sizeof(SpheresGeom), SpheresGeomVars, -1);
  owlGeomTypeSetIntersectProg(SpheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(SpheresGeomType, module, "Spheres");

  owlBuildPrograms(context);
  LOG_OK("BUILD prog DONE\n");

  OWLBuffer frameBuffer = owlManagedMemoryBufferCreate(
      context, OWL_USER_TYPE(neighbors[0]), neighbors.size(), neighbors.data());

  OWLBuffer SpheresBuffer = owlDeviceBufferCreate(
      context, OWL_USER_TYPE(Spheres[0]), Spheres.size(), Spheres.data());

  // ------------------------------------------------------------------
  // create actual geometry
  // ------------------------------------------------------------------

  OWLGeom SpheresGeom = owlGeomCreate(context, SpheresGeomType);
  owlGeomSetPrimCount(SpheresGeom, Spheres.size());
  owlGeomSetBuffer(SpheresGeom, "prims", SpheresBuffer);
  owlGeomSet1f(SpheresGeom, "rad", radius);

  // ##################################################################
  // Launch Params
  // ##################################################################

  OWLVarDecl myGlobalsVars[] = {
      {"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, frameBuffer)},
      {"k", OWL_INT, OWL_OFFSETOF(MyGlobals, k)},
      {"spheres", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, spheres)},
      {"distRadius", OWL_FLOAT, OWL_OFFSETOF(MyGlobals, distRadius)},
      {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);
  owlParamsSetBuffer(lp, "frameBuffer", frameBuffer);
  owlParamsSet1i(lp, "k", knn);
  owlParamsSetBuffer(lp, "spheres", SpheresBuffer);
  owlParamsSet1f(lp, "distRadius", radius);
  LOG_OK("Geoms and Params DONE\n");

  // ------------------------------------------------------------------
  // set up all accel(s) we need to trace into those groups
  // ------------------------------------------------------------------

  OWLGeom userGeoms[] = {SpheresGeom};

  auto start_b = std::chrono::steady_clock::now();
  OWLGroup spheresGroup = owlUserGeomGroupCreate(context, 1, userGeoms);
  owlGroupBuildAccel(spheresGroup);

  OWLGroup world = owlInstanceGroupCreate(context, 1, &spheresGroup);
  owlGroupBuildAccel(world);

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
      //{"spheres",       OWL_BUFPTR, OWL_OFFSETOF(RayGenData,spheres)},
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

  // compute camera frame:
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
  // owlRayGenSetBuffer(rayGen,"spheres",        SpheresBuffer);

  owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
  owlRayGenSetGroup(rayGen, "world", world);
  owlRayGenSet3f(rayGen, "camera.org", (const owl3f &)origin);
  owlRayGenSet3f(rayGen, "camera.llc", (const owl3f &)lower_left_corner);
  owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f &)horizontal);
  owlRayGenSet3f(rayGen, "camera.vert", (const owl3f &)vertical);

  // ------------------------------------------------------------------
  // build shader binding table required to trace the groups
  // ------------------------------------------------------------------
  LOG("building SBT ...");
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);
  LOG_OK("everything set up ...");

  bool foundKNN = 0;
  int numRounds = 0, stop = 0;

  const Neigh *fb;

  // long long int num_intersections = 0;

  auto start = std::chrono::steady_clock::now();

  auto round_start = std::chrono::steady_clock::now();
  auto round_end = std::chrono::steady_clock::now();
  // auto round_elapsed = std::chrono::steady_clock::now();
  ////////////////////////////////////////////////////Call-1////////////////////////////////////////////////////////////////////////////
  while (!foundKNN) //&& (stop==0 | stop==1))
  {
    cout << "\n================================================================"
            "===============================================\nRound: "
         << ++numRounds << " Radius = " << radius << '\n';

    round_start = std::chrono::steady_clock::now();
    cudaDeviceSynchronize();
    owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
    cudaDeviceSynchronize();
    fb = (const Neigh *)owlBufferGetPointer(frameBuffer, 0);

    foundKNN = 1;
    /* Counting number of points left after each round
    // count = 0;
    // for (int j = 0; j < Spheres.size(); j++) {
    //   num_intersections += fb[j * knn].intersections;
    //   if (fb[j * knn].numNeighbors > 0)
    //     count++;
    // }
    // std::cout << "POINTS LEFT: " << count
    //           << "\nNUM INTERSECTIONS: " << num_intersections << '\n';
		*/

    // Determine if we need another round

    for (int j = 0; j < Spheres.size(); j++) {

			/* Uncomment to print neighbors
      // outfile<<"Point "<<j<<": ("<<Spheres.at(j).center.x<<",
      // "<<Spheres.at(j).center.y<<", "<<Spheres.at(j).center.z<<")\n"; 
			// for(int i = 0; i < knn; i++)
      // outfile<<j<<","<<fb[j*knn+i].ind<<','<<fb[j*knn+i].dist<<'\n';
      // cout<<"HOST: numNeighbors["<<j*knn<<"] =
      // "<<fb[j*knn].numNeighbors<<'\n';
			*/
		
      if (fb[j * knn].numNeighbors > 0) {
        foundKNN = 0;
        radius *= 2;
        owlGeomSet1f(SpheresGeom, "rad", radius);
        owlParamsSet1f(lp, "distRadius", radius);
        owlGroupRefitAccel(spheresGroup);
        owlGroupRefitAccel(world);
        break;
      }
    }

    // Time per round

    round_end = std::chrono::steady_clock::now();
    auto round_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        round_end - round_start);
    std::cout << "Round: " << numRounds
              << " Time: " << round_elapsed.count() / 1000000.0 << " seconds"
              << '\n';
  }
  auto end = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "True KNN time: " << elapsed.count() / 1000000.0 << " seconds."
            << std::endl;
  auto tot = elapsed_b.count() / 1000000.0 + elapsed.count() / 1000000.0;
  std::cout << "Total time: " << tot << '\n';

  // Write time taken to file
  std::ofstream outfile;
  outfile.open(argv[6], std::ios::app);
  if (!outfile.is_open()) {
    perror("Error open");
    exit(EXIT_FAILURE);
  }
  outfile << tot << std::endl;

  // ##################################################################
  // and finally, clean up
  // ##################################################################

  owlContextDestroy(context);
}