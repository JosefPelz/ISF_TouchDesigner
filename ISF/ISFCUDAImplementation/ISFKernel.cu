#include "ISFTOP.h"
#include "ISF.h"

#include <cstdio>
#include <iostream>

constexpr auto PI = 3.14159265358979323846264338327950288419;

texture<float4, cudaTextureType2D, cudaReadModeElementType> inTex;
surface<void, cudaSurfaceType2D> outputSurface;


//Conversion from 3D vertex index to 1d index
__device__ int
voxelToId(int x, int y, int z, int3 M) {
	dim3 N = dim3 (M.x, M.y, M.z);
	return (x % N.x) * (N.y * N.z) + (y % N.y) * N.z + (z % N.z);
}

//calculate position from vertex
__device__ float3
voxelToPos(int x, int y, int z, int3 N, float3 L) {
	return make_float3(float(x) / float(N.x) * L.x, float(y) / float(N.y) * L.y, float(z) / float(N.z) * L.z);
}

//complex exponential function
__device__ cuComplex
cexp(float alpha) {
	return make_cuComplex(cos(alpha), sin(alpha));
}

//calculate length of complex number
__device__ float
length(cuComplex z) {
	return sqrt(z.x * z.x + z.y * z.y);
}

//dot product of real-valued 3-vectors
__device__ float
dot(float3 v, float3 w) {
	return v.x * w.x + v.y * w.y + v.z * w.z;
}


//complex multiplication
__device__ cuComplex
cmult(cuComplex v, cuComplex w) {
	return make_cuComplex(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x);
}

//read values of Psi from input texture
__global__ void
readPsi(ISF isf, int width)
{ 
	//calculating unique 3D identifier/vertex of CUDA thread
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	//calculated unique 1d index, corresponding to offset of C++ arrays
	int id = voxelToId(x, y, z, isf.N);


	if (x < isf.N.x && y < isf.N.y && z< isf.N.z) {
		//calculating respective pixel position for index
		int xc = id % width;
		int yc = id / width;

		float4 values = tex2D(inTex, xc, yc);
		isf.Psi1[id] = make_cuComplex(values.x, values.y);	
		isf.Psi2[id] = make_cuComplex(values.z, values.w);
	}
}

//Same as above, but writing to prescribed velocity array
__global__ void
readPrescribedVelocity(ISF isf, int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		int xc = id % width;
		int yc = id / width;

		isf.PrescribedVel[id] = tex2D(inTex, xc, yc);
	}
}


//write values of Psi to output texture/ CUDA surface
__global__ void
writePsi(ISF isf, int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);
	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		//calculating pixel position from index
		int xc = id%width;
		int yc = id/width;
		float psix = isf.Psi1[id].x;
		float psiy = isf.Psi1[id].y;
		float psiz = isf.Psi2[id].x;
		float psiw = isf.Psi2[id].y;
		surf2Dwrite(make_float4(psix, psiy, psiz, psiw), outputSurface, (int)sizeof(float4) * xc, yc);
	}
}


//Schrödinger Flow in Fourier Domain
__global__ void
schroedingerFlow(ISF isf)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		//Offsetted k because Fourier transformation is not centered
		float kx = -(int(float(isf.N.x) / 2. + x) % isf.N.x - float(isf.N.x) / 2.) / isf.L.x;
		float ky = -(int(float(isf.N.y) / 2. + y) % isf.N.y - float(isf.N.y) / 2.) / isf.L.y;
		float kz = -(int(float(isf.N.z) / 2. + z) % isf.N.z - float(isf.N.z) / 2.) / isf.L.z;

		float lambda = - isf.Dt * 2 * PI * PI * isf.Hbar * (kx*kx + ky*ky + kz*kz);

		cuComplex p = cexp(lambda);
		
		isf.Psi1[id] = cmult(isf.Psi1[id], p);
		isf.Psi2[id] = cmult(isf.Psi2[id], p);
	}
}

//normalizing Psi
__global__ void
normalize(ISF isf)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {

		float4 oldPsi = make_float4(isf.Psi1[id].x, isf.Psi1[id].y,
									isf.Psi2[id].x, isf.Psi2[id].y);

		float l = sqrt(	oldPsi.x * oldPsi.x +
						oldPsi.y * oldPsi.y +
						oldPsi.z * oldPsi.z +
						oldPsi.w * oldPsi.w);

		isf.Psi1[id] = make_cuComplex(oldPsi.x/l, oldPsi.y/l);
		isf.Psi2[id] = make_cuComplex(oldPsi.z/l, oldPsi.w/l);
	}
}


//Needed factor after FFT and IFFT.
__global__ void
normalizeFFT(ISF isf)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		float l = float(isf.N.x * isf.N.y * isf.N.z);
		isf.Div[id] = make_cuComplex(isf.Div[id].x / l, isf.Div[id].y / l);
	}
}

//calculation of velocity 1-form
__global__ void
velocityOneForm(ISF isf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		float4 psi = make_float4(isf.Psi1[id].x, isf.Psi1[id].y, isf.Psi2[id].x, isf.Psi2[id].y);
		float3 w;
		
		int id_p = voxelToId(x + 1, y, z, isf.N);
		float4 qsi = make_float4(isf.Psi1[id_p].x, isf.Psi1[id_p].y, isf.Psi2[id_p].x, isf.Psi2[id_p].y);
		float a = psi.x * qsi.x + psi.y * qsi.y + psi.z * qsi.z + psi.w * qsi.w;
		float b = psi.x * qsi.y - psi.y * qsi.x + psi.z * qsi.w - psi.w * qsi.z;
		w.x = atan2(b, a);
		
		id_p = voxelToId(x, y + 1, z, isf.N);
		qsi = make_float4(isf.Psi1[id_p].x, isf.Psi1[id_p].y, isf.Psi2[id_p].x, isf.Psi2[id_p].y);
		a = psi.x * qsi.x + psi.y * qsi.y + psi.z * qsi.z + psi.w * qsi.w;
		b = psi.x * qsi.y - psi.y * qsi.x + psi.z * qsi.w - psi.w * qsi.z;
		w.y = atan2(b, a);

		id_p = voxelToId(x, y, z + 1, isf.N);
		qsi = make_float4(isf.Psi1[id_p].x, isf.Psi1[id_p].y, isf.Psi2[id_p].x, isf.Psi2[id_p].y);
		a = psi.x * qsi.x + psi.y * qsi.y + psi.z * qsi.z + psi.w * qsi.w;
		b = psi.x * qsi.y - psi.y * qsi.x + psi.z * qsi.w - psi.w * qsi.z;
		w.z = atan2(b, a);

		isf.Vel[id] = make_float3(w.x, w.y, w.z);
	}
}

//calculation of divergence
__global__ void
divergence(ISF isf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		float3 w = isf.Vel[id];
		
		int id_m = voxelToId(x - 1, y, z, isf.N);
		float f = (w.x - isf.Vel[id_m].x) / (isf.L.x / float(isf.N.x) * isf.L.x / float(isf.N.x));
		
		id_m = voxelToId(x, y - 1, z, isf.N);
		f += (w.y - isf.Vel[id_m].y) / (isf.L.y / float(isf.N.y) * isf.L.y / float(isf.N.y));
		id_m = voxelToId(x, y, z - 1, isf.N);
		f += (w.z - isf.Vel[id_m].z) / (isf.L.z / float(isf.N.z) * isf.L.z / float(isf.N.z));
		
		isf.Div[id] = make_cuComplex(f, 0);
	}
}

//Solve poisson equation in Fourier domain
__global__ void
poissonSolve(ISF isf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		cufftComplex d = isf.Div[id];
		float sx = sin(PI * x / float(isf.N.x)) / isf.L.x * float(isf.N.x);
		float sy = sin(PI * y / float(isf.N.y)) / isf.L.y * float(isf.N.y);
		float sz = sin(PI * z / float(isf.N.z)) / isf.L.z * float(isf.N.z);
		float denom = sx * sx + sy * sy + sz * sz;
		float fac = 0.0;
		if (x != 0 || y != 0 || z != 0) {
			fac = -0.25 / denom;
		}
		isf.Div[id] = make_cuComplex(fac * d.x, fac * d.y);
	}
}


__global__ void
gaugeTransform(ISF isf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;


	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		float d = isf.Div[id].x;

		cuComplex fac = cexp(-d);
		isf.Psi1[id] = cmult(isf.Psi1[id],fac);
		isf.Psi2[id] = cmult(isf.Psi2[id],fac);
	}
}


//Application of prescribed velocity
__global__ void
applyPrescribedVelocity(ISF isf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {
		float4 V = isf.PrescribedVel[id];
		float3 k = make_float3(V.x / isf.Hbar, V.y / isf.Hbar, V.z / isf.Hbar);
		cuComplex psi1 = isf.Psi1[id];
		cuComplex psi2 = isf.Psi2[id];
		float r1 = length(psi1);
		float r2 = length(psi2);

		float omega = dot(make_float3(V.x,V.y,V.z), make_float3(V.x, V.y, V.z))/(isf.Hbar*2.0);

		float phase = dot(k, voxelToPos(x, y, z, isf.N, isf.L))-omega*isf.T;

		cuComplex fac = cexp(phase);
		isf.Psi1[id].x = (1 - V.w) * psi1.x + V.w * r1 * fac.x;
		isf.Psi1[id].y = (1 - V.w) * psi1.y + V.w * r1 * fac.y;
		isf.Psi2[id].x = (1 - V.w) * psi2.x + V.w * r2 * fac.x;
		isf.Psi2[id].y = (1 - V.w) * psi2.y + V.w * r2 * fac.y;
	}
}


//application of external force
__global__ void
applyExternalForce(ISF isf, float3 ExternalForce) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int id = voxelToId(x, y, z, isf.N);

	if (x < isf.N.x && y < isf.N.y && z < isf.N.z) {

		float phase = dot(voxelToPos(x, y, z, isf.N, isf.L),ExternalForce)*isf.Dt/isf.Hbar;
		cuComplex fac = cexp(phase);
		isf.Psi2[id] = cmult(fac,isf.Psi2[id]);
	}
}


extern "C" void
launch_readPsi(dim3 numBlocks, dim3 threadsPerBlock, cudaArray *g_data_array, ISF isf) {
	cudaBindTextureToArray(inTex, g_data_array);
	//width of corresponding 2d texture
	int width = int(sqrt(float(isf.N.x * isf.N.y * isf.N.z)));
	readPsi <<< numBlocks, threadsPerBlock >>>(isf, width);
}

extern "C" void
launch_readPrescribedVelocity(dim3 numBlocks, dim3 threadsPerBlock, cudaArray * g_data_array, ISF isf) {
	cudaBindTextureToArray(inTex, g_data_array);
	//width of corresponding 2d texture
	int width = int(sqrt(float(isf.N.x * isf.N.y * isf.N.z)));
	readPrescribedVelocity << < numBlocks, threadsPerBlock >> > (isf,width);
}

extern "C" void
launch_writePsi(dim3 numBlocks, dim3 threadsPerBlock, cudaArray *output, ISF isf) {
	cudaBindSurfaceToArray(outputSurface, output);
	//width of corresponding 2d texture
	int width = int(sqrt(float(isf.N.x * isf.N.y * isf.N.z)));
	writePsi <<< numBlocks, threadsPerBlock >>>(isf,width);
}

extern "C" void
launch_ApplicationOfPrescribedVelocity(dim3 numBlocks, dim3 threadsPerBlock, ISF isf) {
	applyPrescribedVelocity << < numBlocks, threadsPerBlock >> > (isf);
}


extern "C" void
launch_ApplicationOfExternalForce(dim3 numBlocks, dim3 threadsPerBlock, ISF isf, float3 ExternalForce) {
	applyExternalForce << < numBlocks, threadsPerBlock >> >(isf, ExternalForce) ;
}


//Algorithm 2: Schrödinger Flow
extern "C" void
launch_schroedingerFlow(dim3 numBlocks, dim3 threadsPerBlock, ISF isf) {
	//perform forward FFT for both components of Psi
	if (cufftExecC2C(isf.Plan, isf.Psi1, isf.Psi1, CUFFT_FORWARD) != cudaSuccess) {
		fprintf(stderr, "CUFFT error: Failed to execute forward C2C isf.Psi1\n");
		return;
	}
	if (cufftExecC2C(isf.Plan, isf.Psi2, isf.Psi2, CUFFT_FORWARD) != cudaSuccess) {
		fprintf(stderr, "CUFFT error: Failed to execute forward C2C isf.Psi2\n");
		return;
	}

	schroedingerFlow << < numBlocks, threadsPerBlock >> > (isf);

	//inverse FFT for both components of Psi
	if (cufftExecC2C(isf.Plan, isf.Psi1, isf.Psi1, CUFFT_INVERSE) != cudaSuccess) {
		fprintf(stderr, "CUFFT error: Failed to execute inverse C2C isf.Psi1\n");
		return;
	}
	if (cufftExecC2C(isf.Plan, isf.Psi2, isf.Psi2, CUFFT_INVERSE) != cudaSuccess) {
		fprintf(stderr, "CUFFT error: Failed to execute inverse C2C isf.Psi2\n");
		return;
	};

	normalize <<< numBlocks, threadsPerBlock >>> (isf);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize schroedinger flow\n");
		return;
	}
}

//Algorithm 3: Pressure projection
extern "C" void
launch_pressureProjection(dim3 numBlocks, dim3 threadsPerBlock, ISF isf){
	velocityOneForm << < numBlocks, threadsPerBlock >> > (isf);
	
	divergence << < numBlocks, threadsPerBlock >> > (isf);
	
	//forward FFT of divergence
	if (cufftExecC2C(isf.Plan, isf.Div, isf.Div, CUFFT_FORWARD) != cudaSuccess) {
		fprintf(stderr, "CUFFT error: Failed to execute forward C2C isf.Div\n");
		return;
	}

	poissonSolve<< < numBlocks, threadsPerBlock >> > (isf);


	//inverse FFT of divergence
	if (cufftExecC2C(isf.Plan, isf.Div, isf.Div, CUFFT_INVERSE) != cudaSuccess) {
		fprintf(stderr, "CUFFT error: Failed to execute inverse C2C isf.Div\n");
		return;
	}
	
	//apply normalization factor of FFT
	normalizeFFT << < numBlocks, threadsPerBlock >> > (isf);
	
	gaugeTransform << < numBlocks, threadsPerBlock >> > (isf);
		
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize pressure projection\n");
		return;
	}
}
