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
#define DENOM_EPS 1e-8f
#define ROUGHNESS_EPS 0.0001f
#define WHITE make_float3(1.f, 1.f, 1.f)

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

rtTextureSampler<float4, 2> Kd_map;
rtDeclareVariable(float2, Kd_map_scale, , );

struct Material
{
	__device__ __inline__ Material()
	{
		base_color = make_float3(0.6f, 0.035f, 0.024f);
		metallic = 1.0f;
		subsurface = 0.0f;
		specular = 0.0f;
		roughness = 0.0f;
		specular_tint = 0.0f;
		anisotropy = 0.0f;
		sheen = 0.0f;
		sheen_tint = 0.0f;
		clearcoat = 0.0f;
		clearcoat_gloss = 0.0f;
	}

	float3 base_color;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specular_tint;
	float anisotropy;
	float sheen;
	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;
};

static __device__ __inline__ float SchlickFresnelReflectance(float u)
{
	float m = clamp(1.f - u, 0.f, 1.f);
	float m2 = m * m;
	return m2 * m2 * m;
}

static __device__ __inline__ float GTR1(float ndoth, float a)
{
	if (a >= 1.f) return 1.f / PI;

	float a2 = a * a;
	float t = 1.f + (a2 - 1.f) * ndoth * ndoth;
	return (a2 - 1.f) / (PI * log(a2) * t);
}

static __device__ __inline__ float GTR2_Aniso(float ndoth, float hdotx, float hdoty, float ax, float ay)
{
	float hdotxa2 = (hdotx / ax);
	hdotxa2 *= hdotxa2;
	float hdotya2 = (hdoty / ay);
	hdotya2 *= hdotya2;
	float denom = hdotxa2 + hdotya2 + ndoth * ndoth;
	return denom > 1e-5 ? (1.f / (PI * ax * ay * denom * denom)) : 0.f;
}


static __device__ __inline__ float SmithGGX_G(float ndotv, float a)
{
	float a2 = a * a;
	float b = ndotv * ndotv;
	return 1.f / (ndotv + sqrtf(a2 + b - a2 * b));
}

static __device__ __inline__ float SmithGGX_G_Aniso(float ndotv, float vdotx, float vdoty, float ax, float ay)
{
	float vdotxax2 = (vdotx * ax) * (vdotx * ax);
	float vdotyay2 = (vdoty * ay) * (vdoty * ay);
	return 1.f / (ndotv + sqrtf(vdotxax2 + vdotyay2 + ndotv * ndotv));
}


static __device__ __inline__ float Disney_GetPdf(Material &mat, float3 wi, float3 wo)
{
	float aspect = sqrtf(1.f - mat.anisotropy * 0.9f);

	float ax = max(0.001f, mat.roughness * mat.roughness * (1.f + mat.anisotropy));
	float ay = max(0.001f, mat.roughness * mat.roughness * (1.f - mat.anisotropy));
	float3 wh = normalize(wo + wi);
	float ndotwh = fabs(wh.y);
	float hdotwo = fabs(dot(wh, wo));

	float d_pdf = fabs(wo.y) / PI;
	float r_pdf = GTR2_Aniso(ndotwh, wh.x, wh.z, ax, ay) * ndotwh / (4.f * hdotwo);
	float c_pdf = GTR1(ndotwh, lerp(0.1f, 0.001f, mat.clearcoat_gloss)) * ndotwh / (4.f * hdotwo);

	float3 cd_lin = mat.base_color;//make_float3(powf(mat.base_color.x, 2.2f), powf(mat.base_color.y, 2.2f), powf(mat.base_color.z, 2.2f));
	// Luminance approximmation
	float cd_lum = dot(cd_lin, make_float3(0.3f, 0.6f, 0.1f));

	// Normalize lum. to isolate hue+sat
	float3 c_tint = cd_lum > 0.f ? (cd_lin / cd_lum) : WHITE;

	float3 c_spec0 = lerp(mat.specular * 0.1f * lerp(WHITE,
		c_tint, mat.specular_tint),
		cd_lin, mat.metallic);

	float cs_lum = dot(c_spec0, make_float3(0.3f, 0.6f, 0.1f));

	float cs_w = cs_lum / (cs_lum + (1.f - mat.metallic) * cd_lum);

	return c_pdf * mat.clearcoat + (1.f - mat.clearcoat) * (cs_w * r_pdf + (1.f - cs_w) * d_pdf);
}


static __device__ __inline__ float3 Disney_Evaluate(Material &mat, float3 wi, float3 wo)
{

	float ndotwi = fabs(wi.y);
	float ndotwo = fabs(wo.y);

	float3 h = normalize(wi + wo);
	float ndoth = fabs(h.y);
	float hdotwo = fabs(dot(h, wo));

	float3 cd_lin = mat.base_color;//make_float3(powf(mat.base_color.x, 2.2f), powf(mat.base_color.y, 2.2f), powf(mat.base_color.z, 2.2f));
	// Luminance approximmation
	float cd_lum = dot(cd_lin, make_float3(0.3f, 0.6f, 0.1f));

	// Normalize lum. to isolate hue+sat
	float3 c_tint = cd_lum > 0.f ? (cd_lin / cd_lum) : WHITE;

	float3 c_spec0 = lerp(mat.specular * 0.1f * lerp(WHITE,
		c_tint, mat.specular_tint),
		cd_lin, mat.metallic);

	float3 c_sheen = lerp(WHITE, c_tint, mat.sheen_tint);

	// Diffuse fresnel - go from 1 at normal incidence to 0.5 at grazing
	// and lerp in diffuse retro-reflection based on mat.roughness
	float f_wo = SchlickFresnelReflectance(ndotwo);
	float f_wi = SchlickFresnelReflectance(ndotwi);

	float fd90 = 0.5f + 2 * hdotwo * hdotwo * mat.roughness;
	float fd = lerp(1.f, fd90, f_wo) * lerp(1.f, fd90, f_wi);

	// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
	// 1.25 scale is used to (roughly) preserve albedo
	// fss90 used to "flatten" retroreflection based on mat.roughness
	float fss90 = hdotwo * hdotwo * mat.roughness;
	float fss = lerp(1.f, fss90, f_wo) * lerp(1.f, fss90, f_wi);
	float ss = 1.25f * (fss * (1.f / (ndotwo + ndotwi) - 0.5f) + 0.5f);

	// mat.specular
	float ax = max(0.001f, mat.roughness * mat.roughness * (1.f + mat.anisotropy));
	float ay = max(0.001f, mat.roughness * mat.roughness * (1.f - mat.anisotropy));
	float ds = GTR2_Aniso(ndoth, h.x, h.z, ax, ay);
	float fh = SchlickFresnelReflectance(hdotwo);
	float3 fs = lerp(c_spec0, WHITE, fh);

	float gs;
	gs = SmithGGX_G_Aniso(ndotwo, wo.x, wo.z, ax, ay);
	gs *= SmithGGX_G_Aniso(ndotwi, wi.x, wi.z, ax, ay);

	// mat.sheen
	float3 f_sheen = fh * mat.sheen * c_sheen;

	// mat.clearcoat (ior = 1.5 -> F0 = 0.04)
	float dr = GTR1(ndoth, lerp(0.1f, 0.001f, mat.clearcoat_gloss));
	float fr = lerp(0.04f, 1.f, fh);
	float gr = SmithGGX_G(ndotwo, 0.25f) * SmithGGX_G(ndotwi, 0.25f);

	return ((1.f / PI) * lerp(fd, ss, mat.subsurface) * cd_lin + f_sheen) *
		(1.f - mat.metallic) + gs * fs * ds + mat.clearcoat * gr * fr * dr;
}


static __device__ __inline__ float3 GetOrthoVector(float3 n)
{
	float3 p;

	if (fabs(n.z) > 0.f) {
		float k = sqrt(n.y*n.y + n.z*n.z);
		p.x = 0; p.y = -n.z / k; p.z = n.y / k;
	}
	else {
		float k = sqrt(n.x*n.x + n.y*n.y);
		p.x = n.y / k; p.y = -n.x / k; p.z = 0;
	}

	return normalize(p);
}


static __device__ __inline__ float3 Sample_MapToHemisphere(float3 n, float e)
{
	// Construct basis
	float2 sample = make_float2(rnd(prd_radiance.seed), rnd(prd_radiance.seed));
	float3 u = GetOrthoVector(n);
	float3 v = cross(u, n);
	u = cross(n, v);

	// Calculate 2D sample
	float r1 = sample.x;
	float r2 = sample.y;

	// Transform to spherical coordinates
	float sinpsi = sin(2 * PI*r1);
	float cospsi = cos(2 * PI*r1);
	float costheta = pow(1.f - r2, 1.f / (e + 1.f));
	float sintheta = sqrt(1.f - costheta * costheta);

	// Return the result
	return normalize(u * sintheta * cospsi + v * sintheta * sinpsi + n * costheta);
}

static __device__ __inline__ float3 Disney_Sample(float3 normal, Material &mat, float3 wi, float3* wo, float* pdf)
{
	float2 sample = make_float2(rnd(prd_radiance.seed), rnd(prd_radiance.seed));
	float ax = max(0.001f, mat.roughness * mat.roughness * (1.f + mat.anisotropy));
	float ay = max(0.001f, mat.roughness * mat.roughness * (1.f - mat.anisotropy));

	float3 wh;


	if (sample.x < mat.clearcoat)
	{
		sample.x /= (mat.clearcoat);

		float a = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
		float ndotwh = sqrtf((1.f - powf(a*a, 1.f - sample.y)) / (1.f - a*a));
		float sintheta = sqrtf(1.f - ndotwh * ndotwh);
		wh = normalize(make_float3(cos(2.f * PI * sample.x) * sintheta,
			ndotwh,
			sin(2.f * PI * sample.x) * sintheta));

		*wo = -wi + 2.f*fabs(dot(wi, wh)) * wh;

	}
	else
	{
		sample.x -= (mat.clearcoat);
		sample.x /= (1.f - mat.clearcoat);

		float3 cd_lin = mat.base_color;//make_float3(powf(mat.base_color.x, 2.2f), powf(mat.base_color.y, 2.2f), powf(mat.base_color.z, 2.2f));
		// Luminance approximmation
		float cd_lum = dot(cd_lin, make_float3(0.3f, 0.6f, 0.1f));

		// Normalize lum. to isolate hue+sat
		float3 c_tint = cd_lum > 0.f ? (cd_lin / cd_lum) : WHITE;

		float3 c_spec0 = lerp(mat.specular * 0.3f * lerp(WHITE,
			c_tint, mat.specular_tint),
			cd_lin, mat.metallic);

		float cs_lum = dot(c_spec0, make_float3(0.3f, 0.6f, 0.1f));

		float cs_w = cs_lum / (cs_lum + (1.f - mat.metallic) * cd_lum);

		if (sample.y < cs_w)
		{
			sample.y /= cs_w;

			float t = sqrtf(sample.y / (1.f - sample.y));
			wh = normalize(make_float3(t * ax * cos(2.f * PI * sample.x),
				1.f,
				t * ay * sin(2.f * PI * sample.x)));

			*wo = -wi + 2.f*fabs(dot(wi, wh)) * wh;
		}
		else
		{
			sample.y -= cs_w;
			sample.y /= (1.f - cs_w);

			*wo = Sample_MapToHemisphere(make_float3(0.0f,1.0f,0.0f), 1.f);

			wh = normalize(*wo + wi);
		}
	}

	//float ndotwh = fabs(wh.y);
	//float hdotwo = fabs(dot(wh, *wo));

	*pdf = Disney_GetPdf(mat, wi, *wo);

	return Disney_Evaluate(mat, wi, *wo);
}

RT_PROGRAM void closest_hit_radiance()
{
	Material mat;
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 out;
	float pdf;
	const optix::Onb onb(ffnormal);

	float3 f = Disney_Sample(ffnormal, mat, -ray.direction, &out, &pdf);
	//onb.inverse_transform(out);

	// update throughput with primitive reflectance
	if (pdf > 0.0f)
	{
		prd_radiance.reflectance *= f * clamp(dot(ffnormal, out), 0.0f, 1.0f) / pdf;
		// update path direction
		prd_radiance.origin = front_hit_point;
		prd_radiance.direction = out;
	}
	else
		prd_radiance.done = true;
}
