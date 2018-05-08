/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixPathTracer: A path tracer using the disney brdf.
//
//-----------------------------------------------------------------------------

#ifndef __APPLE__
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#  endif
#endif

#include <GLFW/glfw3.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "commonStructs.h"
#include "sceneLoader.h"
#include "light_parameters.h"
#include "properties.h"
#include <Camera.h>
#include <OptiXMesh.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixPathTracer";

const int NUMBER_OF_BRDF_INDICES = 2;
const int NUMBER_OF_LIGHT_INDICES = 2;
optix::Buffer m_bufferBRDFSample;
optix::Buffer m_bufferBRDFEval;
optix::Buffer m_bufferBRDFPdf;

optix::Buffer m_bufferLightSample;
optix::Buffer m_bufferMaterialParameters;
optix::Buffer m_bufferLightParameters;

double elapsedTime = 0;
double lastTime = 0;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------
Properties properties;
Context      context = 0;


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

static std::string ptxPath( const std::string& cuda_file )
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}

optix::GeometryInstance createSphere(optix::Context context,
	optix::Material material,
	float3 center,
	float radius)
{
	optix::Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("sphere_intersect.cu");
	sphere->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	sphere->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "sphere_intersect_robust"));

	sphere["center"]->setFloat(center);
	sphere["radius"]->setFloat(radius);

	optix::GeometryInstance instance = context->createGeometryInstance(sphere, &material, &material + 1);
	return instance;
}

optix::GeometryInstance createQuad(optix::Context context,
	optix::Material material,
	float3 v1, float3 v2, float3 anchor, float3 n)
{
	optix::Geometry quad = context->createGeometry();
	quad->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("quad_intersect.cu");
	quad->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	quad->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "intersect"));

	float3 normal = normalize(cross(v1, v2));
	float4 plane = make_float4(normal, dot(normal, anchor));
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	quad["v1"]->setFloat(v1);
	quad["v2"]->setFloat(v2);
	quad["anchor"]->setFloat(anchor);
	quad["plane"]->setFloat(plane);

	optix::GeometryInstance instance = context->createGeometryInstance(quad, &material, &material + 1);
	return instance;
}


static Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}

void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void createContext( bool use_pbo )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );

    // Note: this sample does not need a big stack size even with high ray depths, 
    // because rays are not shot recursively.
    context->setStackSize( 800 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 3 );
    context["cutoff_color"]->setFloat( 0.0f, 0.0f, 0.0f );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-3f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, properties.width, properties.height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Accumulation buffer
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT4, properties.width, properties.height);
    context["accum_buffer"]->set( accum_buffer );

    // Ray generation program
    std::string ptx_path( ptxPath( "path_trace_camera.cu" ) );
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXFile( ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    ptx_path = ptxPath( "background.cu" );
    context->setMissProgram( 0, context->createProgramFromPTXFile( ptx_path, "miss" ) );
	const std::string texture_filename = std::string(sutil::samplesDir()) + "/data/CedarCity.hdr";
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(1.0f)));

	Program prg;
	// BRDF sampling functions.
    m_bufferBRDFSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
    int* brdfSample = (int*) m_bufferBRDFSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
    prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Sample");
    brdfSample[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Sample");
	brdfSample[1] = prg->getId();
    m_bufferBRDFSample->unmap();
    context["sysBRDFSample"]->setBuffer(m_bufferBRDFSample);
	
	// BRDF Eval functions.
    m_bufferBRDFEval = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
    int* brdfEval = (int*) m_bufferBRDFEval->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
    prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Eval");
    brdfEval[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Eval");
	brdfEval[1] = prg->getId();
    m_bufferBRDFEval->unmap();
    context["sysBRDFEval"]->setBuffer(m_bufferBRDFEval);
	
	// BRDF Pdf functions.
    m_bufferBRDFPdf = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
    int* brdfPdf = (int*) m_bufferBRDFPdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
    prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Pdf");
    brdfPdf[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Pdf");
	brdfPdf[1] = prg->getId();
    m_bufferBRDFPdf->unmap();
    context["sysBRDFPdf"]->setBuffer(m_bufferBRDFPdf);

	// Light sampling functions.
	m_bufferLightSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_LIGHT_INDICES);
	int* lightsample = (int*)m_bufferLightSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "sphere_sample");
	lightsample[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "quad_sample");
	lightsample[1] = prg->getId();
	m_bufferLightSample->unmap();
	context["sysLightSample"]->setBuffer(m_bufferLightSample);
}

Material createMaterial(const MaterialParameter &mat, int index)
{
    const std::string ptx_path = ptxPath( "hit_program.cu" );
    Program ch_program = context->createProgramFromPTXFile( ptx_path, "closest_hit" );
	Program ah_program = context->createProgramFromPTXFile(ptx_path, "any_hit");
	
    Material material = context->createMaterial();
    material->setClosestHitProgram( 0, ch_program );
	material->setAnyHitProgram(1, ah_program);
	
	material["materialId"]->setInt(index);
	material["programId"]->setInt(mat.brdf);

    return material;
}

Material createLightMaterial(const LightParameter &mat, int index)
{
	const std::string ptx_path = ptxPath("light_hit_program.cu");
	Program ch_program = context->createProgramFromPTXFile(ptx_path, "closest_hit");

	Material material = context->createMaterial();
	material->setClosestHitProgram(0, ch_program);

	material["lightMaterialId"]->setInt(index);

	return material;
}

void updateMaterialParameters(const std::vector<MaterialParameter> &materials)
{
	MaterialParameter* dst = static_cast<MaterialParameter*>(m_bufferMaterialParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < materials.size(); ++i, ++dst) {
		MaterialParameter mat = materials[i];
		
		dst->color = mat.color;
		dst->emission = mat.emission;
		dst->metallic = mat.metallic;
		dst->subsurface = mat.subsurface;
		dst->specular = mat.specular;
		dst->specularTint = mat.specularTint;
		dst->roughness = mat.roughness;
		dst->anisotropic = mat.anisotropic;
		dst->sheen = mat.sheen;
		dst->sheenTint = mat.sheenTint;
		dst->clearcoat = mat.clearcoat;
		dst->clearcoatGloss = mat.clearcoatGloss;
		dst->brdf = mat.brdf;
	}
	m_bufferMaterialParameters->unmap();
}

void updateLightParameters(const std::vector<LightParameter> &lightParameters)
{
	LightParameter* dst = static_cast<LightParameter*>(m_bufferLightParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < lightParameters.size(); ++i, ++dst) {
		LightParameter mat = lightParameters[i];

		dst->position = mat.position;
		dst->emission = mat.emission;
		dst->radius = mat.radius;
		dst->area = mat.area;
		dst->v1 = mat.v1;
		dst->v2 = mat.v2;
		dst->normal = mat.normal;
		dst->lightType = mat.lightType;
	}
	m_bufferLightParameters->unmap();
}

optix::Aabb createGeometry(
        const std::vector<std::string>& filenames,
        const std::vector<optix::Matrix4x4>& xforms, 
        const std::vector<MaterialParameter> &materials,
		const std::vector<LightParameter> &lights,
        // output: this is a Group with two GeometryGroup children, for toggling visibility later
        optix::Group& top_group
        )
{

    const std::string ptx_path = ptxPath( "triangle_mesh.cu" );

    top_group = context->createGroup();
    top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

    int num_triangles = 0;
	size_t i;
    optix::Aabb aabb;
    {
        GeometryGroup geometry_group = context->createGeometryGroup();
        geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
        top_group->addChild( geometry_group );
		
        for (i = 0; i < filenames.size(); ++i) {
            OptiXMesh mesh;
            mesh.context = context;
            
            // override defaults
            mesh.intersection = context->createProgramFromPTXFile( ptx_path, "mesh_intersect_refine" );
            mesh.bounds = context->createProgramFromPTXFile( ptx_path, "mesh_bounds" );
            mesh.material = createMaterial(materials[i], i);

            loadMesh( filenames[i], mesh, xforms[i] ); 
            geometry_group->addChild( mesh.geom_instance );

            aabb.include( mesh.bbox_min, mesh.bbox_max );

            std::cerr << filenames[i] << ": " << mesh.num_triangles << std::endl;
            num_triangles += mesh.num_triangles;
        }
        std::cerr << "Total triangle count: " << num_triangles << std::endl;
    }
	//Lights
	{
		GeometryGroup geometry_group = context->createGeometryGroup();
		geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
		top_group->addChild(geometry_group);
		for (i = 0; i < lights.size(); ++i)
		{
			GeometryInstance instance;
			if (lights[i].lightType == QUAD)
				instance = createQuad(context, createLightMaterial(lights[i], i), lights[i].v1, lights[i].v2, lights[i].position, lights[i].normal);
			else if (lights[i].lightType == SPHERE)
				instance = createSphere(context, createLightMaterial(lights[i], i), lights[i].position, lights[i].radius);
			geometry_group->addChild(instance);
		}
	}

    context[ "top_object" ]->set( top_group ); 

    return aabb;
}

//------------------------------------------------------------------------------
//
//  GLFW callbacks
//
//------------------------------------------------------------------------------

struct CallbackData
{
    sutil::Camera& camera;
    unsigned int& accumulation_frame;
};

void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    bool handled = false;

    if( action == GLFW_PRESS )
    {
        switch( key )
        {
            case GLFW_KEY_Q:
            case GLFW_KEY_ESCAPE:
                if( context )
                    context->destroy();
                if( window )
                    glfwDestroyWindow( window );
                glfwTerminate();
                exit(EXIT_SUCCESS);

            case( GLFW_KEY_S ):
            {
                const std::string outputImage = std::string(SAMPLE_NAME) + ".png";
                std::cerr << "Saving current frame to '" << outputImage << "'\n";
                sutil::writeBufferToFile( outputImage.c_str(), getOutputBuffer() );
                handled = true;
                break;
            }
            case( GLFW_KEY_F ):
            {
               CallbackData* cb = static_cast<CallbackData*>( glfwGetWindowUserPointer( window ) );
               cb->camera.reset_lookat();
               cb->accumulation_frame = 0;
               handled = true;
               break;
            }
        }
    }

    if (!handled) {
        // forward key event to imgui
        ImGui_ImplGlfw_KeyCallback( window, key, scancode, action, mods );
    }
}

void windowSizeCallback( GLFWwindow* window, int w, int h )
{
    if (w < 0 || h < 0) return;

    const unsigned width = (unsigned)w;
    const unsigned height = (unsigned)h;

    CallbackData* cb = static_cast<CallbackData*>( glfwGetWindowUserPointer( window ) );
    if ( cb->camera.resize( width, height ) ) {
        cb->accumulation_frame = 0;
    }

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glViewport(0, 0, width, height);
}


//------------------------------------------------------------------------------
//
// GLFW setup and run 
//
//------------------------------------------------------------------------------

GLFWwindow* glfwInitialize( )
{
    GLFWwindow* window = sutil::initGLFW();

    // Note: this overrides imgui key callback with our own.  We'll chain this.
    glfwSetKeyCallback( window, keyCallback );

    glfwSetWindowSize( window, (int)properties.width, (int)properties.height);
    glfwSetWindowSizeCallback( window, windowSizeCallback );

    return window;
}


void glfwRun( GLFWwindow* window, sutil::Camera& camera, const optix::Group top_group )
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, properties.width, properties.height);

    unsigned int frame_count = 0;
    unsigned int accumulation_frame = 0;
    float transmittance_log_scale = 0.0f;
    int max_depth = 3;
	lastTime = sutil::currentTime();

    // Expose user data for access in GLFW callback functions when the window is resized, etc.
    // This avoids having to make it global.
    CallbackData cb = { camera, accumulation_frame };
    glfwSetWindowUserPointer( window, &cb );

    while( !glfwWindowShouldClose( window ) )
    {

        glfwPollEvents();                                                        

        ImGui_ImplGlfw_NewFrame();

        ImGuiIO& io = ImGui::GetIO();
        
        // Let imgui process the mouse first
        if (!io.WantCaptureMouse) {

            double x, y;
            glfwGetCursorPos( window, &x, &y );

            if ( camera.process_mouse( (float)x, (float)y, ImGui::IsMouseDown(0), ImGui::IsMouseDown(1), ImGui::IsMouseDown(2) ) ) {
                accumulation_frame = 0;
            }
        }

        // imgui pushes
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,   ImVec2(0,0) );
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha,          0.6f        );
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f        );

		
        sutil::displayFps( frame_count++ );
		sutil::displaySpp( accumulation_frame );

        {
            static const ImGuiWindowFlags window_flags = 
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_AlwaysAutoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoScrollbar;

            ImGui::SetNextWindowPos( ImVec2( 2.0f, 70.0f ) );
            ImGui::Begin("controls", 0, window_flags );
            if ( ImGui::CollapsingHeader( "Controls", ImGuiTreeNodeFlags_DefaultOpen ) ) {
                if (ImGui::SliderInt( "max depth", &max_depth, 1, 10 )) {
                    context["max_depth"]->setInt( max_depth );
                    accumulation_frame = 0;
                }
            }
            ImGui::End();
        }

		elapsedTime += sutil::currentTime() - lastTime;
		if (accumulation_frame == 0)
			elapsedTime = 0;
		sutil::displayElapsedTime(elapsedTime);
		lastTime = sutil::currentTime();

        // imgui pops
        ImGui::PopStyleVar( 3 );

        // Render main window
        context["frame"]->setUint( accumulation_frame++ );
        context->launch( 0, camera.width(), camera.height() );
        sutil::displayBufferGL( getOutputBuffer() );

        // Render gui over it
        ImGui::Render();

        glfwSwapBuffers( window );
    }
    
    destroyContext();
    glfwDestroyWindow( window );
    glfwTerminate();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options] [mesh0 mesh1 ...]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help                  Print this usage message and exit.\n"
        "  -f | --file <output_file>    Save image to file and exit.\n"
        "  -n | --nopbo                 Disable GL interop for display buffer.\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".png'\n"
        "  f  Re-center camera\n"
        "\n"
        "Mesh files are optional and can be OBJ or PLY.\n"
        << std::endl;

    exit(1);
}


int main( int argc, char** argv )
{
    bool use_pbo  = true;
    std::string scene_file;
	std::string out_file;
    std::vector<std::string> mesh_files;
    std::vector<optix::Matrix4x4> mesh_xforms;
	std::vector<MaterialParameter> materials;
	std::vector<LightParameter> lights;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
		else if (arg == "-scene")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			scene_file = argv[++i];
			LoadScene(scene_file.c_str(), mesh_files, mesh_xforms, materials, lights, properties);
		}
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if( arg[0] == '-' )
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
        else {
            // Interpret argument as a mesh file.
            mesh_files.push_back( argv[i] );
            mesh_xforms.push_back( optix::Matrix4x4::identity() );
        }
    }

    try
    {
		if (mesh_files.empty()) {

			// Default scene
			scene_file = sutil::samplesDir() + std::string("/data/spaceship.scene");
			LoadScene(scene_file.c_str(), mesh_files, mesh_xforms, materials, lights, properties);
		}

        GLFWwindow* window = glfwInitialize();

#ifndef __APPLE__
        GLenum err = glewInit();
        if (err != GLEW_OK)
        {
            std::cerr << "GLEW init failed: " << glewGetErrorString( err ) << std::endl;
            exit(EXIT_FAILURE);
        }
#endif

        createContext( use_pbo );

		m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		m_bufferLightParameters->setElementSize(sizeof(LightParameter));
		m_bufferLightParameters->setSize(lights.size());
		updateLightParameters(lights);
		context["sysLightParameters"]->setBuffer(m_bufferLightParameters);
		
		m_bufferMaterialParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		m_bufferMaterialParameters->setElementSize(sizeof(MaterialParameter));
		m_bufferMaterialParameters->setSize(materials.size());
		updateMaterialParameters(materials);
		context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);

		context["sysNumberOfLights"]->setInt(lights.size());
        optix::Group top_group;
		const optix::Aabb aabb = createGeometry(mesh_files, mesh_xforms, materials, lights, top_group);

        context->validate();

        const optix::float3 camera_eye( optix::make_float3( 0.0f, 1.5f*aabb.extent( 1 ), 1.5f*aabb.extent( 2 ) ) );
		//const optix::float3 camera_eye(optix::make_float3(-12.08, 11.68, -0.432));
        const optix::float3 camera_lookat( aabb.center() );
        const optix::float3 camera_up( optix::make_float3( 0.0f, 1.0f, 0.0f ) );
        sutil::Camera camera( properties.width, properties.height, 
                &camera_eye.x, &camera_lookat.x, &camera_up.x,
                context["eye"], context["U"], context["V"], context["W"] );

        if ( out_file.empty() )
        {
            glfwRun( window, camera, top_group );
        }
        else
        {
            // Accumulate frames for anti-aliasing
            const unsigned int numframes = 256;
            std::cerr << "Accumulating " << numframes << " frames ..." << std::endl;
            for ( unsigned int frame = 0; frame < numframes; ++frame ) {
                context["frame"]->setUint( frame );
                context->launch( 0, properties.width, properties.height );
            }
            sutil::writeBufferToFile( out_file.c_str(), getOutputBuffer() );
            std::cerr << "Wrote " << out_file << std::endl;
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

