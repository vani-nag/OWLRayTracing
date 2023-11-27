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

// The Ray Tracing in One Weekend scene, but with cubes substituted for some
// spheres. This program shows how different geometric types in a single scene
// are handled.

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "GeomTypes.h"
// viewer base class, for window and user interaction
#include "owlViewer/OWLViewer.h"

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

const vec2i init_fbSize(1600,800);
const vec3f init_lookFrom(13, 2, 3);
const vec3f init_lookAt(0, 0, 0);
const vec3f init_lookUp(0.f,1.f,0.f);
const float init_fovy = 20.f;

std::vector<DielectricSphere> dielectricSpheres;
std::vector<LambertianSphere> lambertianSpheres;
std::vector<MetalSphere>      metalSpheres;




void createScene()
{
  lambertianSpheres.push_back({Sphere{vec3f(0.f, -1.0f, -1.f), 1000.f},
        Lambertian{vec3f(0.5f, 0.5f, 0.5f)}});
  dielectricSpheres.push_back({Sphere{vec3f(0.f, 1.f, 0.f), 1.f},
        Dielectric{1.5f}});
  lambertianSpheres.push_back({Sphere{vec3f(-4.f,1.f, 0.f), 1.f},
        Lambertian{vec3f(0.4f, 0.2f, 0.1f)}});
  metalSpheres.push_back({Sphere{vec3f(4.f, 1.f, 0.f), 1.f},
        Metal{vec3f(0.7f, 0.6f, 0.5f), 0.0f}});
}
  




struct Viewer : public owl::viewer::OWLViewer
{
  Viewer();
  
  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;
  
      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */     
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;
  
  OWLRayGen  rayGen  { 0 };
  OWLContext context { 0 };
  OWLGroup   world   { 0 };
  OWLBuffer  accumBuffer { 0 };
  int        accumID     { 0 };
};


/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
  const vec3f lookFrom = camera.getFrom();
  const vec3f lookAt = camera.getAt();
  const vec3f lookUp = camera.getUp();
  const float cosFovy = camera.getCosFovy();
  const float vfov = owl::viewer::toDegrees(acosf(cosFovy));
  // ........... compute variable values  ..................
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

  accumID = 0;
  
  // ----------- set variables  ----------------------------
  owlRayGenSetGroup (rayGen,"world",        world);
  owlRayGenSet3f    (rayGen,"camera.org",   (const owl3f&)origin);
  owlRayGenSet3f    (rayGen,"camera.llc",   (const owl3f&)lower_left_corner);
  owlRayGenSet3f    (rayGen,"camera.horiz", (const owl3f&)horizontal);
  owlRayGenSet3f    (rayGen,"camera.vert",  (const owl3f&)vertical);
}

void Viewer::render()
{
  owlRayGenSet1i(rayGen,"accumID",accumID);
  accumID++;
  owlBuildSBT(context);
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
}


/*! window notifies us that we got resized */     
void Viewer::resize(const vec2i &newSize)
{
  OWLViewer::resize(newSize);
  cameraChanged();
  
  if (accumBuffer)
    owlBufferResize(accumBuffer,newSize.x*newSize.y*sizeof(float4));
  else
    accumBuffer = owlDeviceBufferCreate(context,OWL_FLOAT4,
                                        newSize.x*newSize.y,nullptr);
  
  owlRayGenSetBuffer(rayGen,"accumBuffer",  accumBuffer);
  owlRayGenSet1ul   (rayGen,"fbPtr",        (uint64_t)fbPointer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);

}

Viewer::Viewer()
  : OWLViewer("RTOW on OWL (mixed geometries)",
              init_fbSize)
{
  // ##################################################################
  // init owl
  // ##################################################################

  context = owlContextCreate(nullptr,1);
  OWLModule  module  = owlModuleCreate(context,ptxCode);
  
  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare *sphere* geometry type(s)
  // -------------------------------------------------------

  // ----------- metal -----------
  OWLVarDecl metalSpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(MetalSpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType metalSpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(MetalSpheresGeom),
                        metalSpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(metalSpheresGeomType,0,
                           module,"MetalSpheres");
  owlGeomTypeSetIntersectProg(metalSpheresGeomType,0,
                              module,"MetalSpheres");
  owlGeomTypeSetBoundsProg(metalSpheresGeomType,
                           module,"MetalSpheres");

  // ----------- dielectric -----------
  OWLVarDecl dielectricSpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(DielectricSpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType dielectricSpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(DielectricSpheresGeom),
                        dielectricSpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(dielectricSpheresGeomType,0,
                           module,"DielectricSpheres");
  owlGeomTypeSetIntersectProg(dielectricSpheresGeomType,0,
                              module,"DielectricSpheres");
  owlGeomTypeSetBoundsProg(dielectricSpheresGeomType,
                           module,"DielectricSpheres");

  // ----------- lambertian -----------
  OWLVarDecl lambertianSpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(LambertianSpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType lambertianSpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(LambertianSpheresGeom),
                        lambertianSpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(lambertianSpheresGeomType,0,
                           module,"LambertianSpheres");
  owlGeomTypeSetIntersectProg(lambertianSpheresGeomType,0,
                              module,"LambertianSpheres");
  owlGeomTypeSetBoundsProg(lambertianSpheresGeomType,
                           module,"LambertianSpheres");



  

  // -------------------------------------------------------
  // make sure to do that *before* setting up the geometry, since the
  // user geometry group will need the compiled bounds programs upon
  // accelBuild()
  // -------------------------------------------------------
  owlBuildPrograms(context);






  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ====================== SPHERES ======================
  
  // ----------- metal -----------
  OWLBuffer metalSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(metalSpheres[0]),
                            metalSpheres.size(),metalSpheres.data());
  OWLGeom metalSpheresGeom
    = owlGeomCreate(context,metalSpheresGeomType);
  owlGeomSetPrimCount(metalSpheresGeom,metalSpheres.size());
  owlGeomSetBuffer(metalSpheresGeom,"prims",metalSpheresBuffer);

  // ----------- lambertian -----------
  OWLBuffer lambertianSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(lambertianSpheres[0]),
                            lambertianSpheres.size(),lambertianSpheres.data());
  OWLGeom lambertianSpheresGeom
    = owlGeomCreate(context,lambertianSpheresGeomType);
  owlGeomSetPrimCount(lambertianSpheresGeom,lambertianSpheres.size());
  owlGeomSetBuffer(lambertianSpheresGeom,"prims",lambertianSpheresBuffer);

  // ----------- dielectric -----------
  OWLBuffer dielectricSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(dielectricSpheres[0]),
                            dielectricSpheres.size(),dielectricSpheres.data());
  OWLGeom dielectricSpheresGeom
    = owlGeomCreate(context,dielectricSpheresGeomType);
  owlGeomSetPrimCount(dielectricSpheresGeom,dielectricSpheres.size());
  owlGeomSetBuffer(dielectricSpheresGeom,"prims",dielectricSpheresBuffer);



  
  
  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  // ----------- one group for the spheres -----------
  /* (note these are user geoms, so have to be in another group than the triangle
     meshes) */
  OWLGeom  userGeoms[] = {
    lambertianSpheresGeom,
    metalSpheresGeom,
    dielectricSpheresGeom
  };
  OWLGroup userGeomGroup
    = owlUserGeomGroupCreate(context,3,userGeoms);
  owlGroupBuildAccel(userGeomGroup);

 
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
    { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(RayGenData,accumBuffer)},
    { "accumID",       OWL_INT,    OWL_OFFSETOF(RayGenData,accumID)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.origin)},
    { "camera.llc",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.lower_left_corner)},
    { "camera.horiz",  OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.horizontal)},
    { "camera.vert",   OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.vertical)},
    { /* sentinel to mark end of list */ }
  };

  // ........... create object  ............................
  rayGen
    = owlRayGenCreate(context,module,"rayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);
  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  // programs have been built before, but have to rebuild raygen and
  // miss progs
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);
}

int main(int ac, char **av)
{
  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

  LOG("owl example '" << av[0] << "' starting up");

  LOG("creating the scene ...");
  createScene();
  LOG_OK("created scene:");
  LOG_OK(" num lambertian spheres: " << lambertianSpheres.size());
  LOG_OK(" num dielectric spheres: " << dielectricSpheres.size());
  LOG_OK(" num metal spheres     : " << metalSpheres.size());

  Viewer viewer;
  viewer.camera.setOrientation(init_lookFrom,
                               init_lookAt,
                               init_lookUp,
                               init_fovy);
  viewer.enableFlyMode();
  viewer.enableInspectMode(/* the big sphere in the middle: */
                           owl::box3f(vec3f(-1,0,-1),vec3f(1,2,1)));
  viewer.showAndRun();
  
  LOG("destroying devicegroup ...");
  owlContextDestroy(viewer.context);
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
