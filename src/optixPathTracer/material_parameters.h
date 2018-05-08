#pragma once

#ifndef MATERIAL_PARAMETER_H
#define MATERIAL_PARAMETER_H

#include"rt_function.h"

enum BrdfType
{
	DISNEY, GLASS
};

struct MaterialParameter
{
	RT_FUNCTION MaterialParameter()
	{
		color = optix::make_float3(0.0f, 0.0f, 0.0f);
		emission = optix::make_float3(0.0f);
		metallic = 0.0;
		subsurface = 0.0f;
		specular = 0.5f;
		roughness = 0.5f;
		specularTint = 0.0f;
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.5f;
		clearcoat = 0.0f;
		clearcoatGloss = 1.0f;
		brdf = DISNEY;
	}

	optix::float3 color;
	optix::float3 emission;
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
	BrdfType brdf;
};

#endif
