#pragma once

#ifndef STATE_H
#define STATE_H

#include "material_parameters.h"
#include "prd.h"

struct State
{
	optix::float3 fhp;
	optix::float3 bhp;
	optix::float3 normal;
	optix::float3 ffnormal;
};

#endif
