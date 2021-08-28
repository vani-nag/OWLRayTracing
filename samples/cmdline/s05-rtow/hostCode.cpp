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
#include<string>
#include<sstream>
#include <random>
#include<ctime>
#include<chrono>
#include<algorithm>
#include<set>



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

const vec3f lookFrom(13, 2, 3);
//const vec3f lookFrom(-2.f,1.f,1.f);
const vec3f lookAt(0, 0, 0);

const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<Sphere> Spheres;
std::vector<int> neighbors;
std::vector<int> seed;
std::vector<DisjointSet> ds;

//DisjointSet find

int find(int x, const DisjointSet *d)
{
	// Finds the representative of the set
	// that x is an element of
	if (d[x].parent != x) {

		// if x is not the parent of itself
		// Then x is not the representative of
		// his set,
	
		//optixLaunchParams.frameBuffer[x].parent = find(optixLaunchParams.frameBuffer[x].parent);
		return(find(d[x].parent,d));
		// so we recursively call Find on its parent
		// and move i's node directly under the
		// representative of this set
	}

	//printf("Leaving find()\n");
	return 	d[x].parent;
}



int main(int ac, char **av)
{
  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

	std::string line;
  std::ifstream myfile;
	myfile.open("/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/3droad.csv");
		
  if(!myfile.is_open())
  {
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
	float radius = 0.01;
	int minPts = 100;

	for(int i = 0, j = 0; i < vect.size(); i+=3, j+=1)
	{
		Spheres.push_back(Sphere{vec3f(vect.at(i),vect.at(i+1),vect.at(i+2)),-1});
		ds.push_back(DisjointSet{j,0,0});
	}
		
	
	//Frame Buffer
	const vec2i fbSize(Spheres.size(),1);

	//createScene();
  //LOG_OK("created scene:");
  LOG_OK(" num spheres: " << Spheres.size());


  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate(nullptr,1);
  OWLModule  module  = owlModuleCreate(context,ptxCode);
	LOG_OK("init DONE\n");
	
	
  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  OWLVarDecl SpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom,prims)},
    { "rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom,rad)},
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
	LOG_OK("BUILD prog DONE\n");
	
	
  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  //LOG("building geometries ...");

  /*OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,
                            Spheres.size());*/
   /*OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_USER_TYPE(ds[0]),
                            Spheres.size());  */                   

	OWLBuffer frameBuffer
    = owlManagedMemoryBufferCreate(context,OWL_USER_TYPE(ds[0]),
                            ds.size(), ds.data());
 
  OWLBuffer SpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(Spheres[0]),
                            Spheres.size(),Spheres.data());
                            
  OWLGeom SpheresGeom
    = owlGeomCreate(context,SpheresGeomType);
  owlGeomSetPrimCount(SpheresGeom,Spheres.size());
  owlGeomSetBuffer(SpheresGeom,"prims",SpheresBuffer);
  owlGeomSet1f(SpheresGeom,"rad",radius);
  
  

  // ##################################################################
  // Params
  // ##################################################################
  
  OWLVarDecl myGlobalsVars[] = {
	{"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, frameBuffer)},
	{"callNum", OWL_INT, OWL_OFFSETOF(MyGlobals, callNum)},
	{"minPts", OWL_INT, OWL_OFFSETOF(MyGlobals,minPts)},
	{ /* sentinel to mark end of list */ }
	};
	
	OWLParams lp = owlParamsCreate(context,sizeof(MyGlobals),myGlobalsVars,-1);
	owlParamsSetBuffer(lp,"frameBuffer",frameBuffer);	
	owlParamsSet1i(lp,"minPts",minPts);	
	
	
	LOG_OK("Geoms and Params DONE\n");
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




		
	// -------------------------------------------------------
	// set up ray gen program
	// -------------------------------------------------------
	OWLVarDecl rayGenVars[] = {
		{ "spheres",       OWL_BUFPTR, OWL_OFFSETOF(RayGenData,spheres)},
		{ "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
		{ "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},	
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
	owlRayGenSetBuffer(rayGen,"spheres",        SpheresBuffer);
	owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
	owlRayGenSetGroup (rayGen,"world",        world);
	owlRayGenSet3f    (rayGen,"camera.org",   (const owl3f&)origin);
	owlRayGenSet3f    (rayGen,"camera.llc",   (const owl3f&)lower_left_corner);
	owlRayGenSet3f    (rayGen,"camera.horiz", (const owl3f&)horizontal);
	owlRayGenSet3f    (rayGen,"camera.vert",  (const owl3f&)vertical);

	// programs have been built before, but have to rebuild raygen and
	// miss progs
	auto start_b = std::chrono::steady_clock::now();
	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);
	auto end_b = std::chrono::steady_clock::now();
	auto elapsed_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
	std::cout << "Build time: " << elapsed_b.count()/1000000.0 << " seconds." << std::endl;

	// ##################################################################
	// DBSCAN Start
	// ##################################################################
		
	std::ofstream ofile;
	ofile.open("/home/min/a/nagara16/Downloads/owl/build/op.txt", std::ios::out);
	ofile.close();
	ofile.open("/home/min/a/nagara16/Downloads/owl/build/op.txt", std::ios::app);
	
	////////////////////////////////////////////////////Call-1////////////////////////////////////////////////////////////////////////////
	/*int *d_x,x[4];
	cudaMalloc(&d_x,4*sizeof(int));
	for(int i=0;i<4;i++)
		x[i]=i;
	cudaMemcpy(x, d_x, 4*sizeof(int), cudaMemcpyHostToDevice);
	
	DeviceMemory d;
	d.alloc(Spheres.size()*sizeof(Sphere));
	d.upload(SpheresBuffer);*/
	
	
	owlParamsSet1i(lp, "callNum", 1);
	auto start = std::chrono::steady_clock::now();
	owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);
	cout<<"Call-1 done\n";
	/*d.download(&Spheres[0]);
	d.free();
	cout<<"Status must be -3: "<<Spheres[0].status<<'\n';*/
	/*const Sphere *sb = (const Sphere*)owlBufferGetPointer(SpheresBuffer,3);
	cout<<"Status must be -3: "<<sb[0].status<<'\n';*/
	/*cout<<"FrameBuffer in HOST\n";
	for(int i = 0; i < sCount; i++)
	{
			//Spheres[i].status = 1;
		cout<<i<<'\t'<<fb[i].parent<<'\t'<<"isCore = "<< fb[i].isCore<<'\n';
	}*/
	
	
	////////////////////////////////////////////////////Call-2////////////////////////////////////////////////////////////////////////////	
	owlParamsSet1i(lp, "callNum", 2);
	owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	const DisjointSet *fb = (const DisjointSet*)owlBufferGetPointer(frameBuffer,0);
	

	std::cout << "Execution time: " << elapsed.count()/1000000.0 << " seconds." << std::endl;
	cout<<"Call-2: FrameBuffer in HOST\n";
	for(int i = 0; i < Spheres.size(); i++)
	{
		ofile << i << '\t'<< find(fb[i].parent,fb)<< std::endl;
		//cout<<i<<'\t'<<find(fb[i].parent,fb)<<'\n';
	}
	
	
	

	
	

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
	//LOG("destroying devicegroup ...");
  owlContextDestroy(context);

  //LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
