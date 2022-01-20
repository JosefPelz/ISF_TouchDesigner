#pragma once

#include "cuda_runtime.h"
#include "cufft.h"

class ISF
{
public:
	ISF();
	void initBuffers();
	void deleteBuffers();

	int3 N;
	float3 L;
	float Hbar;
	float Dt;
	float T;
	bool Flow;
	bool BuffersInitialized;

	cufftComplex* Psi1, * Psi2;
	float3* Vel;
	cufftComplex* Div;
	cufftHandle Plan;
	float4* PrescribedVel;
};