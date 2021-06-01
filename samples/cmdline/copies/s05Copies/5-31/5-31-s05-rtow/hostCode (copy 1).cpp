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
// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include <random>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "s05-rtow.png";
//const vec2i fbSize(1600,800);
const vec2i fbSize(1,1);
const vec3f lookFrom(13, 2, 3);
//const vec3f lookFrom(-2.f,1.f,1.f);
const vec3f lookAt(0, 0, 0);

const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<Sphere> Spheres;


inline float rnd()
{
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

/*void createScene()
{
	Spheres.push_back(Sphere{vec3f(0.f, 0.0f, 0.f), 1.f, 0});
	  
	Spheres.push_back(Sphere{vec3f(-5.f, 0.f, 0.f), 1.f, 1});

	Spheres.push_back(Sphere{vec3f(-4.f,0.f, 0.f), 1.f, 2});

	Spheres.push_back(Sphere{vec3f(1.f, 0.f, 0.f), 1.f, 3});

}*/


	
  
int main(int ac, char **av)
{
  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

  LOG("owl example '" << av[0] << "' starting up");

  LOG("creating the scene ...");

	/////////////////////////////////////////////////////INPUTS//////////////////////////////////////////////////////
  std::string line;
    std::ifstream myfile;
    myfile.open("/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/input.csv");

   if(!myfile.is_open()) {
      perror("Error open");
      exit(EXIT_FAILURE);
   }
   std::vector<float> vect;
    while(getline(myfile, line)) {
     //std::cout << line << '\n';
    std::stringstream ss(line);

    float i;

    while (ss >> i)
    {
        vect.push_back(i);
				//std::cout << i <<'\n';
        if (ss.peek() == ',')
            ss.ignore();
    }
    }	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Select ray origin
  vec3f rayOrigin = vec3f(0.139151609324475f,	0.028264654206815f, -1.f);
	float radius = 0.01;


  //createScene(vect);
	for(int i = 0; i < vect.size(); i+=2)
		 Spheres.push_back(Sphere{vec3f(vect.at(i),vect.at(i+1),0),radius,i});
  LOG_OK("created scene:");
  LOG_OK(" num spheres: " << Spheres.size());


  
  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate(nullptr,1);
  OWLModule  module  = owlModuleCreate(context,ptxCode);
  
  
  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type(s)
  // -------------------------------------------------------

  // ----------- metal -----------
  OWLVarDecl SpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };


  OWLGeomType SpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(SpheresGeom),
                        SpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(SpheresGeomType,0,
                           module,"Spheres");
  owlGeomTypeSetIntersectProg(SpheresGeomType,0,
                              module,"Spheres");
  owlGeomTypeSetBoundsProg(SpheresGeomType,
                           module,"Spheres");

  // make sure to do that *before* setting up the geometry, since the
  // user geometry group will need the compiled bounds programs upon
  // accelBuild()
  owlBuildPrograms(context);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

 
  OWLBuffer SpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(Spheres[0]),
                            Spheres.size(),Spheres.data());
  OWLGeom SpheresGeom
    = owlGeomCreate(context,SpheresGeomType);
  owlGeomSetPrimCount(SpheresGeom,Spheres.size());
  owlGeomSetBuffer(SpheresGeom,"prims",SpheresBuffer);

 

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
  
  OWLGeom  userGeoms[] = {
    SpheresGeom
  };

  OWLGroup spheresGroup
    = owlUserGeomGroupCreate(context,1,userGeoms);
  owlGroupBuildAccel(spheresGroup);
  
  OWLGroup world
    = owlInstanceGroupCreate(context,1,&spheresGroup);
  owlGroupBuildAccel(world);
  // ##################################################################
  // set miss and raygen programs
  // ##################################################################
  
  // -------------------------------------------------------
  // set up miss prog 
  // -------------------------------------------------------
  OWLVarDecl missProgVars[] = {
    { /* sentinel to mark end of list */ }
  };
  // ........... create object  ............................
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  owlMissProgSet(context,0,missProg);
  
  // ........... set variables  ............................
  /* nothing to set */

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "origin",				 OWL_FLOAT3, OWL_OFFSETOF(RayGenData,origin)},	
    { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.origin)},
    { "camera.llc",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.lower_left_corner)},
    { "camera.horiz",  OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.horizontal)},
    { "camera.vert",   OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.vertical)},
    { /* sentinel to mark end of list */ }
  };

  // ........... create object  ............................
  OWLRayGen rayGen
    = owlRayGenCreate(context,module,"rayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

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
  const vec3f lower_left_corner
    = origin - half_width * focusDist*u - half_height * focusDist*v - focusDist * w;
  const vec3f horizontal = 2.0f*half_width*focusDist*u;
  const vec3f vertical = 2.0f*half_height*focusDist*v;


  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSetGroup (rayGen,"world",        world);
  owlRayGenSet3f    (rayGen,"origin",   		(const owl3f&)rayOrigin);
  owlRayGenSet3f    (rayGen,"camera.org",   (const owl3f&)origin);
  owlRayGenSet3f    (rayGen,"camera.llc",   (const owl3f&)lower_left_corner);
  owlRayGenSet3f    (rayGen,"camera.horiz", (const owl3f&)horizontal);
  owlRayGenSet3f    (rayGen,"camera.vert",  (const owl3f&)vertical);
  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  // programs have been built before, but have to rebuild raygen and
  // miss progs
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  
  LOG("launching ...");
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);

  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb
    = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
