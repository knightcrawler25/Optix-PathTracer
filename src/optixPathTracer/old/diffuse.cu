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

rtDeclareVariable(float3, var_color, , );
rtDeclareVariable(float, var_metallic, , );
rtDeclareVariable(float, var_subsurface, , );
rtDeclareVariable(float, var_specular, , );
rtDeclareVariable(float, var_roughness, , );
rtDeclareVariable(float, var_specularTint, , );
rtDeclareVariable(float, var_anisotropic, , );
rtDeclareVariable(float, var_sheen, , );
rtDeclareVariable(float, var_sheenTint, , );
rtDeclareVariable(float, var_clearcoat, , );
rtDeclareVariable(float, var_clearcoatGloss, , );

rtTextureSampler<float4, 2> Kd_map;
rtDeclareVariable( float2, Kd_map_scale, , );

struct Material
{
	__device__ __inline__ Material()
	{
		color = var_color;
		metallic = var_metallic;
		subsurface = var_subsurface;
		specular = var_specular;
		roughness = var_roughness;
		specularTint = var_specularTint;
		anisotropic = var_anisotropic;
		sheen = var_sheen;
		sheenTint = var_sheenTint;
		clearcoat = var_clearcoat;
		clearcoatGloss = var_clearcoatGloss;
	}

	float3 color;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
};


static __device__ __inline__ float sqr(float x) { return x*x; }

static __device__ __inline__ float SchlickFresnel(float u)
{
    float m = clamp(1.0f-u, 0.0f, 1.0f);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

static __device__ __inline__ float GTR1(float NDotH, float a)
{
    if (a >= 1.0f) return (1.0f/PI);
    float a2 = a*a;
    float t = 1.0f + (a2-1.0f)*NDotH*NDotH;
    return (a2-1.0f) / (PI*logf(a2)*t);
}

static __device__ __inline__ float GTR2(float NDotH, float a)
{
    float a2 = a*a;
    float t = 1.0f + (a2-1.0f)*NDotH*NDotH;
    return a2 / (PI * t*t);
}

static __device__ __inline__ float GTR2_aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
{
    return 1.0f / ( PI * ax*ay * sqr( sqr(HDotX/ax) + sqr(HDotY/ay) + NDotH*NDotH ));
}

static __device__ __inline__ float smithG_GGX(float NDotv, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NDotv*NDotv;
    return 1.0f/(NDotv + sqrtf(a + b - a*b));
}

static __device__ __inline__ float BRDFPdf(const Material &mat, const float3& P, const float3& n, const float3& V, const float3& L)
{
    const float a = max(0.001f, mat.roughness);
	float b = lerp(0.1f, 0.001f, mat.clearcoatGloss);
	float ratio = 1.0f / (1.0f + mat.clearcoat);
	float diffuseRatio = 0.5f * (1.f - mat.metallic);
	float specularRatio = 1.f - diffuseRatio;

	const float3 half = normalize(L+V);

	const float cosThetaHalf = abs(dot(half, n));
    const float pdfGTR2 = GTR2(cosThetaHalf, a) * cosThetaHalf;
	const float pdfGTR1 = GTR1(cosThetaHalf, b) * cosThetaHalf;

	// calculate pdf for each method given outgoing light vector
	float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));

    float pdfDiff = abs(dot(L, n))* (1.0f / PI);

    // weight pdfs according to roughness
	//return lerp(pdfSpec, pdfDiff, mat.roughness);
	return diffuseRatio * pdfDiff + specularRatio * pdfSpec;

}

// generate an importance sampled brdf direction
static __device__ __inline__ float3 BRDFSample(const Material &mat, const float3& P, const float3& V, const float3&n)
{
    float3 light;
	
    const float select = rnd(prd_radiance.seed);

	const float r1 = rnd(prd_radiance.seed);
	const float r2 = rnd(prd_radiance.seed);
	const optix::Onb onb( n );

	float diffuseRatio = 0.5f * (1.f - mat.metallic);

	if (select < diffuseRatio) //roughness
    {
        // sample diffuse
		cosine_sample_hemisphere( r1, r2, light );
		onb.inverse_transform(light);
    }
    else
    {
		const float a = max(0.001f, mat.roughness);

        const float phiHalf = r1 * 2.0f * PI;
        
        const float cosThetaHalf = sqrtf((1.0f-r2)/(1.0f + (sqr(a)-1.0f)*r2));      
        const float sinThetaHalf = sqrtf(max(0.0f,1.0f-sqr(cosThetaHalf)));
        const float sinPhiHalf = sinf(phiHalf);
        const float cosPhiHalf = cosf(phiHalf);

		float3 half = make_float3(sinThetaHalf*sinPhiHalf, sinThetaHalf*cosPhiHalf, cosThetaHalf);
		onb.inverse_transform(half);

        light = 2.0f*dot(V, half)*half - V;

    }
	return light;
}


static __device__ __inline__ float3 BRDFEval(const Material &mat, const float3& P, const float3& N, const float3& V, const float3& L)
{
	float NDotL = dot(N, L);
	float NDotV = dot(N, V);
	if (NDotL <= 0.0f || NDotV <= 0.0f) 
		return make_float3(0.0f);

	float3 H = normalize(L + V);
	float NDotH = dot(N, H);
	float LDotH = dot(L, H);

	float3 Cdlin = mat.color;
	float Cdlum = 0.3f*Cdlin.x + 0.6f*Cdlin.y + 0.1f*Cdlin.z; // luminance approx.

	float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
	float3 Cspec0 = lerp(mat.specular*0.08f*lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
	float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
	float Fd90 = 0.5f + 2.0f * LDotH*LDotH * mat.roughness;
	float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

	// Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LDotH*LDotH*mat.roughness;
	float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

	// specular
	//float aspect = sqrt(1-mat.anisotrokPic*.9);
	//float ax = Max(.001f, sqr(mat.roughness)/aspect);
	//float ay = Max(.001f, sqr(mat.roughness)*aspect);
	//float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);
	float a = max(0.001f, mat.roughness);
	float Ds = GTR2(NDotH, a);
	float FH = SchlickFresnel(LDotH);
	float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
	float roughg = sqr(mat.roughness*0.5f + 0.5f);
	float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

	// sheen
	float3 Fsheen = FH * mat.sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
	float Fr = lerp(0.04f, 1.0f, FH);
	float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

	float3 out = ((1.0f / PI) * lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
		* (1.0f - mat.metallic)
		+ Gs*Fs*Ds + 0.25f*mat.clearcoat*Gr*Fr*Dr;

    return out;
}

RT_PROGRAM void closest_hit_radiance()
{
	Material mat;
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 brdfDir = BRDFSample(mat, front_hit_point, -ray.direction, ffnormal);

	float brdfPdf = BRDFPdf(mat, front_hit_point, ffnormal, -ray.direction, brdfDir);

	// reflectance
	float3 f = BRDFEval(mat, front_hit_point, ffnormal, -ray.direction, brdfDir);

	// update throughput with primitive reflectance
	if (brdfPdf > 0.0f)
	{
		prd_radiance.reflectance *= f * clamp(dot(ffnormal,brdfDir), 0.0f, 1.0f) / brdfPdf;
		// update path direction
		prd_radiance.origin = front_hit_point;
		prd_radiance.direction = brdfDir;	
	}
	//else
	//	prd_radiance.done = true;
}
