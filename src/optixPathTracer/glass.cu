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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu_matrix_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "rt_function.h"
#include "material_parameters.h"
#include "state.h"

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// -----------------------------------------------------------------------------

RT_FUNCTION float fresnel( float cos_theta_i, float cos_theta_t, float eta )
{
    const float rs = ( cos_theta_i - cos_theta_t*eta ) / 
                     ( cos_theta_i + eta*cos_theta_t );
    const float rp = ( cos_theta_i*eta - cos_theta_t ) /
                     ( cos_theta_i*eta + cos_theta_t );

    return 0.5f * ( rs*rs + rp*rp );
}

RT_FUNCTION float3 logf( float3 v )
{
    return make_float3( logf(v.x), logf(v.y), logf(v.z) );
}

RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	prd.pdf = 1.0f;
}

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
    const float3 w_out = -ray.direction;
    float3 normal = state.normal;
    float cos_theta_i = optix::dot( w_out, normal );

    float eta;
	float3 transmittance = make_float3(1.0f);
	float3 extinction = -logf(make_float3(0.905f, 0.63f, 0.3));
    if( cos_theta_i > 0.0f ) 
	{
		eta = 1.45f;
    } 
	else 
	{
		transmittance = optix::expf(-extinction * t_hit);
        eta = 1.0f / 1.45f;
        cos_theta_i = -cos_theta_i;
        normal = -normal;
    }
	//intData.mat.color = transmittance;

    float3 w_t;
    const bool tir  = !optix::refract( w_t, -w_out, normal, eta );
    const float cos_theta_t = -optix::dot( normal, w_t );
    const float R  = tir  ? 1.0f : fresnel( cos_theta_i, cos_theta_t, eta );

    const float z = rnd(prd.seed);
    if( z <= R ) 
	{
        // Reflect
		prd.origin = state.fhp;
		prd.direction =  optix::reflect( -w_out, normal );
    } 
	else 
	{
        // Refract
		prd.origin = state.bhp;
		prd.direction = w_t;
    }
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	return mat.color;
}

