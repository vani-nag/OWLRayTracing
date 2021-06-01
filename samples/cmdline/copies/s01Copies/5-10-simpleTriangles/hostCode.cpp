// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

// This program sets up a single geometric object, a mesh for a cube, and
// its acceleration structure, then ray traces it.

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <vector>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<vec3f> circles;  
}triangles;



const int NUM_INDICES = 1;
vec3i indices[NUM_INDICES] =
  {
    {0,1,2}
  };

const char *outFileName = "s01-simpleTriangles.png";
const vec2i fbSize(1,1);
//const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookFrom(13.f,2.f,3.f);
//const vec3f lookFrom(-1.f,-1.f,3.f);
//const vec3f lookFrom(-3.f,1.f,1.f);
//const vec3f lookFrom(-0.73f,3.f,-1.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

void createTrianglesFromSphere(float x, float y, float radius)
{
        float y_pos_max, y_neg_max, x_pos_max, x_neg_max;
        y_pos_max = y + radius;
        y_neg_max = y - radius;

        x_pos_max = x + radius;
        x_neg_max = x - radius;

        //vertices
        std::cout<<x_neg_max<<", "<<y+radius+radius<<", 0"<<"\n";
        std::cout<<x_pos_max+radius+radius<<", "<<y<<", 0"<<"\n";
        std::cout<<x_neg_max<<", "<<y-radius-radius<<", 0"<<"\n\n";

	/*triangles.vertices.push_back(vec3f(x_neg_max, y+radius+radius-1, 0));
	triangles.vertices.push_back(vec3f(x_neg_max, y-radius-radius+1, 0));
	triangles.vertices.push_back(vec3f(x_neg_max+radius+1, y, 0));	*/

	triangles.circles.push_back(vec3f(x,y,0));

	triangles.vertices.push_back(vec3f(x_neg_max, y+radius+radius, 0));//1
	triangles.vertices.push_back(vec3f(x_pos_max+radius+radius, y, 0));//2
	triangles.vertices.push_back(vec3f(x_neg_max, y-radius-radius, 0));//3

	
	const int startIndex = (int)triangles.vertices.size()-3;
	for (int i=0;i<NUM_INDICES;i++)
    		triangles.indices.push_back(indices[i]+vec3i(startIndex));
			
}


int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  // create a context on the first device:
  OWLContext context = owlContextCreate(nullptr,1);
  OWLModule module = owlModuleCreate(context,ptxCode);

  //create triangles
  createTrianglesFromSphere(0.f,2.f,1.f);
  createTrianglesFromSphere(0.75f,2.f,1.f);

  createTrianglesFromSphere(0.74f,1.f,1.f);
  createTrianglesFromSphere(0.73f,1.f,1.f);

  createTrianglesFromSphere(0.72f,1.f,1.f);
  createTrianglesFromSphere(0.53f,1.f,1.f);
  createTrianglesFromSphere(0.50f,1.f,1.f);

  createTrianglesFromSphere(-0.5f,0.f,1.f);
  createTrianglesFromSphere(-0.75f,3.f,1.f);
  createTrianglesFromSphere(-0.74f,0.f,1.f);
  createTrianglesFromSphere(-0.73f,3.f,1.f);

  createTrianglesFromSphere(1.f,1.f,1.f);
  createTrianglesFromSphere(1.1f,1.f,1.f);
  createTrianglesFromSphere(1.12f,1.f,1.f);

  //Select ray origin
  vec3f rayOrigin = vec3f(0.73f, 1.f, -1.f);


  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "circle",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,circle)}
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,3);
  /*owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");*/
  owlGeomTypeSetAnyHit(trianglesGeomType,0, module,"tmesh");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,triangles.vertices.size(),triangles.vertices.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,triangles.indices.size(),triangles.indices.data());
  OWLBuffer circleBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,triangles.circles.size(),triangles.circles.data());
  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);
  
  LOG("No of vertices = " << triangles.vertices.size());
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          triangles.vertices.size(),sizeof(triangles.vertices[0]),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         triangles.indices.size(),sizeof(triangles.indices[0]),0);

  for(int i = 0; i < triangles.indices.size(); i++)
  	printf("index from createTriangle = %d,%d,%d \n",triangles.indices.at(i).x,triangles.indices.at(i).y,triangles.indices.at(i).z );
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
  owlGeomSetBuffer(trianglesGeom,"circle",circleBuffer);
  
  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  OWLGroup trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  OWLGroup world
    = owlInstanceGroupCreate(context,1,&trianglesGroup);
  owlGroupBuildAccel(world);
  

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog 
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  
  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "origin",		OWL_FLOAT3, OWL_OFFSETOF(RayGenData,origin)},	
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
    { /* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  OWLRayGen rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

  // ----------- compute variable values  ------------------
  vec3f camera_pos = lookFrom;
  vec3f camera_d00
    = normalize(lookAt-lookFrom);
  float aspect = fbSize.x / float(fbSize.y);
  vec3f camera_ddu
    = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
  vec3f camera_ddv
    = cosFovy * normalize(cross(camera_ddu,camera_d00));
  camera_d00 -= 0.5f * camera_ddu;
  camera_d00 -= 0.5f * camera_ddv;

  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSetGroup (rayGen,"world",        world);
  owlRayGenSet3f    (rayGen,"origin",   (const owl3f&)rayOrigin);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
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
  assert(fb);
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
