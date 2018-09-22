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
#include "light_parameters.h"
#include "state.h"

using namespace optix;

rtDeclareVariable(int, sysNumberOfLights, , );

RT_FUNCTION float3 UniformSampleSphere(float u1, float u2)
{
	float z = 1.f - 2.f * u1;
	float r = sqrtf(max(0.f, 1.f - z * z));
	float phi = 2.f * M_PIf * u2;
	float x = r * cosf(phi);
	float y = r * sinf(phi);

	return make_float3(x, y, z);
}

RT_CALLABLE_PROGRAM void sphere_sample(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)
{
	const float r1 = rnd(prd.seed);
	const float r2 = rnd(prd.seed);
	sample.surfacePos = light.position + UniformSampleSphere(r1, r2) * light.radius;
	sample.normal = normalize(sample.surfacePos - light.position);
	sample.emission = light.emission * sysNumberOfLights;
}

RT_CALLABLE_PROGRAM void quad_sample(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)
{
	const float r1 = rnd(prd.seed);
	const float r2 = rnd(prd.seed);
	sample.surfacePos = light.position + light.u * r1 + light.v * r2;
	sample.normal = light.normal;
	sample.emission = light.emission * sysNumberOfLights;
}
