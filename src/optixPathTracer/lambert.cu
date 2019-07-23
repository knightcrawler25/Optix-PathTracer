/*
 Copyright Disney Enterprises, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License
 and the following modification to it: Section 6 Trademarks.
 deleted and replaced with:

 6. Trademarks. This License does not grant permission to use the
 trade names, trademarks, service marks, or product names of the
 Licensor and its affiliates, except as required for reproducing
 the content of the NOTICE file.

 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
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


RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	float3 n = state.ffnormal;
	float3 L = prd.bsdfDir;
	
	float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

	prd.pdf =  pdfDiff;

}

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	float3 N = state.ffnormal;
	prd.origin = state.fhp;

	float3 dir;
	
	float r1 = rnd(prd.seed);
	float r2 = rnd(prd.seed);

	optix::Onb onb( N );

	cosine_sample_hemisphere(r1, r2, dir);
	onb.inverse_transform(dir);
	
	prd.bsdfDir = dir;
}


RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	float3 N = state.ffnormal;
	float3 V = prd.wo;
	float3 L = prd.bsdfDir;

	float NDotL = dot(N, L);
	float NDotV = dot(N, V);
	if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

	float3 out = (1.0f / M_PIf) * mat.color;

	return out * clamp(dot(N, L), 0.0f, 1.0f);
}
