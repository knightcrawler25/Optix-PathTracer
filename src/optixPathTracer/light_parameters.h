#pragma once

#ifndef LIGHT_PARAMETER_H
#define LIGHT_PARAMETER_H

#include"rt_function.h"

enum LightType
{
	SPHERE, QUAD
};

struct LightParameter
{
	optix::float3 position;
	optix::float3 normal;
	optix::float3 emission;
	optix::float3 v1;
	optix::float3 v2;
	float area;
	float radius;
	LightType lightType;
};

struct LightSample
{
	optix::float3 surfacePos;
	optix::float3 normal;
	optix::float3 emission;
	float pdf;
	
};

#endif
