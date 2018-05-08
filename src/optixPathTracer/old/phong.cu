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

using namespace optix;

#define PI 3.14159265358979323846f

rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );

rtDeclareVariable(Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

rtTextureSampler<float4, 2> Kd_map;
rtDeclareVariable( float2, Kd_map_scale, , );

struct Material
{
	__device__ __inline__ Material()
	{
		color = make_float3(0.6f,0.3f,0.2f);
		kd = 0.1f;
		ks = 0.9f;
		shininess = 512.0f;
	}

	float3 color;
	float kd;
	float ks;
	float shininess;
};

static __device__ __inline__ float3 BRDFEval(const Material &mat, const float3& P, const float3& N, const float3& V, float3& newdir)
{
	float theta = 0.0f;
	float phi = 0.0f;
	float brdf = 0.0f;
	float pdf = 0.0f;
	float3 reflectionDir = reflect(V, N);
	float u = rnd(prd_radiance.seed);

	// diffuse sample
	if (u < mat.kd) {
		phi = 2.0f * M_PI * rnd(prd_radiance.seed);
		theta = acos(sqrt(rnd(prd_radiance.seed)));
		pdf = (1.0f / M_PI) * cos(theta);
		brdf += (1.0f / M_PI) * mat.kd;
	}

	// specular sample - check the reference to understand where these calculations are derived
	if ((u >= mat.kd) && (u < mat.kd + mat.ks)) {

		theta = acos(pow(rnd(prd_radiance.seed), 1 / (mat.shininess + 1) ));
		if (theta > M_PI * 0.5) 
			theta = M_PI * 0.5;
		phi = 2.0f * M_PI * rnd(prd_radiance.seed);

		pdf = ((mat.shininess + 1) / (2.0f * M_PI)) * pow(cos(theta), mat.shininess);
		brdf = ((mat.shininess + 2) / (2.0f * M_PI)) * pow(cos(theta), mat.shininess) * mat.ks;
	}

	if (u > (mat.ks + mat.kd)) {
		return make_float3(0, 0, 0);
	}

	float3 scy;
	if (u < mat.kd) 
		scy = N;	
	else		
		scy = reflectionDir;

	float3 upv = make_float3(0, 1, 0);
	if (dot(scy, upv) > 0.99f) {
		upv = make_float3(1, 0, 0);
	}

	float3 scx = normalize(cross(upv, scy));
	float3 scz = normalize(cross(scx, scy));

	float ssx = cos(phi) * sin(theta);
	float ssy = cos(theta);
	float ssz = sin(phi) * sin(theta);

	// construct transformation matrix

	Matrix3x3 rotmatrix;
	rotmatrix.setCol(0, scx);
	rotmatrix.setCol(1, scy);
	rotmatrix.setCol(2, scz);

	newdir = rotmatrix * make_float3(ssx, ssy, ssz);

	/*if (dot(newdir, N) < 0) {
		newdir = N;
	}*/

	if (isnan(newdir.x) || isnan(newdir.y) || isnan(newdir.z)) {
		newdir = make_float3(0, 1, 0);
	}

	if (pdf <= 0.0f || brdf <= 0.0f) {
		pdf = 1.0f;
		brdf = 1.0f;
	}

	if (pdf < 0.0001f) {
		pdf = 0.0001f;
	}

	newdir = normalize(newdir);

	float3 mask = (mat.color * dot(N, newdir) * brdf) / pdf;

    return mask;
}

RT_PROGRAM void closest_hit_radiance()
{
	Material mat;
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	// reflectance
	float3 newdir;

	// update throughput with primitive reflectance
	prd_radiance.reflectance *= BRDFEval(mat, front_hit_point, ffnormal, ray.direction, newdir);
	prd_radiance.origin = front_hit_point;
	prd_radiance.direction = newdir;
}
