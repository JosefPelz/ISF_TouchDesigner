#include "ISF.h"
#include <cstdio>

ISF::ISF(){
	//Setting default values on init
	N = make_int3(0,0,0);
	L = make_float3(-1.0, -1.0, -1.0);
	Hbar = 0.0;
	Dt = 0.0;
	T = 0.0;
	Flow = false;
	BuffersInitialized = false;

	Psi1 = nullptr;
	Psi2 = nullptr;
	Vel = nullptr;
	Div = nullptr;
	Plan = cufftHandle();
	PrescribedVel = 0;
}

void ISF::initBuffers() {
	if (N.x != 0) {
		int totalSize = N.x * N.y * N.z;
		
		// Creating cufftHandle Plan for the FFT
		if (cufftPlan3d(&Plan, N.x, N.y, N.z, CUFFT_C2C) != cudaSuccess) {
			fprintf(stderr, "CUFFT error: Failed to create cufft plan 3d\n");
			return;
		}

		//Allocating device memory for all arrays
		cudaMalloc((void**)&Psi1, sizeof(cufftComplex) * totalSize);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to allocate Psi1 \n");
			return;
		}

		cudaMalloc((void**)&Psi2, sizeof(cufftComplex) * totalSize);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to allocate Psi2 \n");
			return;
		}
		
		cudaMalloc((void**)&Div, sizeof(cufftComplex) * totalSize);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to allocate div \n");
			return;
		}
		cudaMalloc((void**)&Vel, sizeof(float3) * totalSize);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to allocate vel \n");
			return;
		}
		cudaMalloc((void**)&PrescribedVel, sizeof(float4) * totalSize);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to allocate externalVel \n");
			return;
		}
		BuffersInitialized = true;
	}
}
void ISF::deleteBuffers(){
	if (BuffersInitialized) {
		//Free device memory
		cudaFree(Psi1);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free Psi1\n");
			return;
		}
		cudaFree(Psi2);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free Psi2\n");
			return;
		}
		cufftDestroy(Plan);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: destroy PsiPlan\n");
			return;
		}
		cudaFree(Div);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free div\n");
			return;
		}
		cudaFree(Vel);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free vel\n");
			return;
		}
		cudaFree(PrescribedVel);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free vel\n");
			return;
		}
		BuffersInitialized = false;
	}
};