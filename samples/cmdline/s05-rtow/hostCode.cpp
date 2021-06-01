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
#include <vector>
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include <random>
#include<ctime>
#include<chrono>

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


/*void createScene()
{
	Spheres.push_back(Sphere{vec3f(0.f, 0.0f, 0.f), 1.f, 0});
	  
	Spheres.push_back(Sphere{vec3f(0.74f, 1.f, 0.f), 1.f, 1});

	Spheres.push_back(Sphere{vec3f(0.73f, 1.f, 0.f), 1.f, 2});

	Spheres.push_back(Sphere{vec3f(-0.74f, 1.f, 0.f), 1.f, 3});
}*/

int main(int ac, char **av)
{
  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

 		std::string line;
    std::ifstream myfile;
    myfile.open("/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/input.csv");
		
   if(!myfile.is_open()) {
      perror("Error open");
      exit(EXIT_FAILURE);
   }
   std::vector<float> vect;
    while(getline(myfile, line)) 
		{
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


  // ##################################################################
  // Create scene
  // ##################################################################

	//Select minPts,epsilon
	float radius = 1.f;
	int minPts = 3;
	int c = 0;

	for(int i = 0; i < vect.size(); i+=3)
	{
		Spheres.push_back(Sphere{vec3f(vect.at(i),vect.at(i+1),vect.at(i+2)),radius,c,0});
		c++;
	}


	//createScene();
  //LOG_OK("created scene:");
  //LOG_OK(" num spheres: " << Spheres.size());


  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate(nullptr,1);
  OWLModule  module  = owlModuleCreate(context,ptxCode);


  // ##################################################################
  // Params
  // ##################################################################

  OWLVarDecl paramVars[] = {
    { "indices",  OWL_BUFPTR, OWL_OFFSETOF(param,indices)},
    { /* sentinel to mark end of list */ }
  };
  OWLBuffer lp
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(param),
                            sizeof(param),0);
	//OWLParams  lp = owlParamsCreate(context,sizeof(param),paramVars,-1);
	
  
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
  /*owlGeomTypeSetClosestHit(SpheresGeomType,0,
                           module,"Spheres");*/
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

  //LOG("building geometries ...");

  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,vect.size());

 
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
    //LOG("building accels ...");
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

	int cluster_number = 0, flag = 0;
	std::ofstream ofile;
	//ofile.open("/home/min/a/nagara16/Downloads/owl/build/cir_op.txt", std::ios::out);
	auto start = std::chrono::steady_clock::now();
  // ##################################################################
  // Start LOOP Here????
  // ##################################################################
	for (auto it = Spheres.begin(); it != Spheres.end(); ++it)
	{
		cout<<"Status = "<<it->status<<'\n';
		if(it -> status == 0)
		{
			//Select ray origin
			vec3f rayOrigin = it -> center;
			cout<<"Spheres: "<<"\t\n";
			for(int i = 0; i < Spheres.size(); i++)
				cout<<Spheres.at(i).center.x<<", "<<Spheres.at(i).center.y<<", "<<Spheres.at(i).center.z<<'\n';
			//cout<<" \tOrigin: "<<rayOrigin.x<<", "<<rayOrigin.y<<", "<<rayOrigin.z<<'\n';


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
			
			//LOG("launching ...");

			owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);

			// ##################################################################
			// Write to file
			// ##################################################################  
			
			ofile.open("/home/min/a/nagara16/Downloads/owl/build/cir_op.txt", std::ios::app);
			const uint32_t *fb
				= (const uint32_t*)owlBufferGetPointer(frameBuffer,0);

				flag = 0;
				for (auto i = Spheres.begin(); i != Spheres.end(); ++i)
				{
					if (fb[i -> index] == 1) 
					{
						ofile << i->index << "," << cluster_number << std::endl;
						cout<<"Erasing "<< i->index<<"\tcluster_number = "<<cluster_number<<'\n';
					    Spheres.erase(i);
					    i--;
						flag = 1;
	      			}
				}

			ofile.close();
			
			//Not core point ==> status = 1 
			if(flag == 0)
				it -> status = 1;
			else
				cluster_number++;
			it--;
		}

	}


  // ##################################################################
  // and finally, clean up
  // ##################################################################
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Execution time: " << elapsed.count()/1000000.0 << " seconds." << std::endl;
  
	//LOG("destroying devicegroup ...");
  owlContextDestroy(context);

  //LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
