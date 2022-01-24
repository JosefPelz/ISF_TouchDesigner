/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */

#include "ISFTOP.h"

#include <assert.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <string.h>
#endif
#include <cstdio>

#include <iostream>


// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{
DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plug1in.
	info->executeMode = TOP_ExecuteMode::CUDA;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case	
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Isfcuda");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("ISF CUDA");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("ISF");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Josef Luis Pelz");
	info->customOPInfo.authorEmail->setString("contact.pelz@gmail.com");

	// This TOP works with 0 or 1 inputs connected
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll

    // Note we can't do any OpenGL work during instantiation

	return new ISFTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL

    // We do some OpenGL teardown on destruction, so ask the TOP_Context
    // to set up our OpenGL context

	delete (ISFTOP*)instance;
}

};


ISFTOP::ISFTOP(const OP_NodeInfo* info, TOP_Context *context)
: myNodeInfo(info), myExecuteCount(0), myError(nullptr)
{
	reset = true;  //Setting reset to true on creation to load initial data
}

ISFTOP::~ISFTOP()
{
	isf.deleteBuffers(); //deallocating memory space when instance is deleted
}

void
ISFTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs *inputs, void* reserved)
{
	// Setting cookEveryFrame to true causes the TOP to cook every frame even
	// if none of its inputs/parameters are changing. Set it to false if it
    // only needs to cook when inputs/parameters change.
	ginfo->cookEveryFrame = false;
	ginfo->cookEveryFrameIfAsked = inputs->getParInt("Cookeveryframe");
	//Only cook every frame if the respective parameter is set to true and the output is used somewhere
}

bool
ISFTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs *inputs, void* reserved)
{
	format->redChannel = true;
	format->greenChannel = true;
	format->blueChannel = true;
	format->alphaChannel = true;

	format->bitsPerChannel = 32;
	format->floatPrecision = true;
	//Setting output texture format to 32-bit rgba
	return true;
}


//Functions defined in the ISFKernel.cu file
extern "C" void
launch_readPsi(dim3 numBlocks, dim3 threadsPerBlock, cudaArray *g_data_array, ISF isf);

extern "C" void
launch_readPrescribedVelocity(dim3 numBlocks, dim3 threadsPerBlock, cudaArray * g_data_array, ISF isf);

extern "C" void
launch_writePsi(dim3 numBlocks, dim3 threadsPerBlock,cudaArray *output, ISF isf);

extern "C" void
launch_schroedingerFlow(dim3 numBlocks, dim3 threadsPerBlock, ISF isf);

extern "C" void
launch_pressureProjection(dim3 numBlocks, dim3 threadsPerBlock, ISF isf);

extern "C" void
launch_ApplicationOfPrescribedVelocity(dim3 numBlocks, dim3 threadsPerBlock, ISF isf);

extern "C" void
launch_ApplicationOfExternalForce(dim3 numBlocks, dim3 threadsPerBlock, ISF isf, float3 ExternalForce);

#include <iostream>
#include <typeinfo>

//This function is executed every time the Cplusplus TOP cooks (is executed)
void
ISFTOP::execute(TOP_OutputFormatSpecs* outputFormat ,
							const OP_Inputs* inputs,
							TOP_Context* context,
							void* reserved)
{
	myExecuteCount++;

	myError = nullptr;

	cudaArray *inputMem = nullptr;
	if (inputs->getNumInputs() > 0) //only execute if any input provided
	{
		const OP_TOPInput* topInput = inputs->getInputTOP(0);
		const OP_TimeInfo* timeInfo = inputs->getTimeInfo();
		if (timeInfo->deltaFrames > 0 || reset) { //We check if deltaFrames>0 to prevent multiple cooks on one frame
			//reading parameters
			int Nx, Ny, Nz;
			inputs->getParInt3("N", Nx, Ny, Nz);
			if (Nx != isf.N.x || Ny != isf.N.y || Nz != isf.N.z) reset = true; //If resolution changed, reset simulation
			isf.N = make_int3(Nx, Ny, Nz);

			double Lx, Ly, Lz;
			inputs->getParDouble3("L", Lx, Ly, Lz);
			isf.L = make_float3(Lx, Ly, Lz);
			isf.Hbar = inputs->getParDouble("Hbar");
			isf.Dt = inputs->getParDouble("Dt");
			int tpbx, tpby, tpbz;
			inputs->getParInt3("Threadsperblock", tpbx, tpby, tpbz);
			isf.Flow = inputs->getParInt("Schroedingerintegration");
			int applyPrescribedForces = inputs->getParInt("Prescribedvelocity");
			int repetitions = inputs->getParInt("Repeatpressureprojection");
			bool loadPrescribedVelocityEachFrame = inputs->getParInt("Loadprescribedvelocityeachframe")==1;
			bool applyExternalForce = inputs->getParInt("Applyexternalforce") == 1;
			double efx,efy,efz;
			inputs->getParDouble3("Externalforce", efx, efy, efz);


			//calculating number of blocks
			dim3 threadsPerBlock(tpbx, tpby, tpbz);
			dim3 numBlocks;
			numBlocks.x = int(ceil(float(Nx) / float(tpbx)));
			numBlocks.y = int(ceil(float(Ny) / float(tpby)));
			numBlocks.z = int(ceil(float(Nz) / float(tpbz)));

			if (topInput->cudaInput == nullptr) //no valid input provided
			{
				myError = "CUDA memory for input TOP was not mapped correctly.";
				return;
			}

			if (topInput->pixelFormat != GL_RGBA32F) { //input is expected to have 32-bit rgba pixel format
				myError = "Input Texture must be 32-bit float (RGBA).";
				return;
			}

			if (reset) { //read from input texture only on reset. Otherwise the last state of Psi will be used
				inputMem = topInput->cudaInput;
				if (isf.BuffersInitialized) isf.deleteBuffers(); //clear buffers before reloading
				isf.initBuffers();	//allocating memory & creating cufftHandle plan
				launch_readPsi(numBlocks, threadsPerBlock, inputMem, isf); //read from input texture
				isf.T = 0.0; //setting total time of simulation to 0
			}

			//Prescribed Velocity from 2nd input if valid and requested by user
			if (applyPrescribedForces && isf.BuffersInitialized && inputs->getNumInputs() > 1 && (reset || loadPrescribedVelocityEachFrame)) {
				launch_readPrescribedVelocity(numBlocks, threadsPerBlock, inputs->getInputTOP(1)->cudaInput, isf);
			}

			if (isf.BuffersInitialized) {
				//User can deactive the Schrödinger integration, e.g., for initial pressure projection.
				if (isf.Flow) {
					launch_schroedingerFlow(numBlocks, threadsPerBlock, isf);
					if (applyExternalForce) launch_ApplicationOfExternalForce(numBlocks, threadsPerBlock, isf, make_float3(efx, efy, efz));
					if (applyPrescribedForces) {
						launch_pressureProjection(numBlocks, threadsPerBlock, isf);
						launch_ApplicationOfPrescribedVelocity(numBlocks, threadsPerBlock, isf);
					}
				}

				for (int i = 0; i < repetitions; i++) {
					if (applyPrescribedForces && !isf.Flow) launch_ApplicationOfPrescribedVelocity(numBlocks, threadsPerBlock, isf);
					launch_pressureProjection(numBlocks, threadsPerBlock, isf);
				}	

				//Write current Psi to output texture
				launch_writePsi(numBlocks, threadsPerBlock, outputFormat->cudaOutput[0], isf);
				//Increase total time by time step
				isf.T += isf.Dt;
			}
			reset = false;
		}
	}
}

int32_t
ISFTOP::getNumInfoCHOPChans(void* reserved)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 1;
}

void
ISFTOP::getInfoCHOPChan(int32_t index,
						OP_InfoCHOPChan* chan,
						void* reserved)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}
}

bool		
ISFTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved)
{
	infoSize->rows = 1;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
ISFTOP::getInfoDATEntries(int32_t index,
										int32_t nEntries,
										OP_InfoDATEntries* entries,
										void* reserved)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
        strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
        snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
ISFTOP::getErrorString(OP_String *error, void* reserved)
{
    error->setString(myError);
}


//Settin up the parameter interface
void
ISFTOP::setupParameters(OP_ParameterManager* manager, void* reserved)
{
	{
		OP_NumericParameter	np;

		np.name = "Reset";
		np.label = "Reset";

		manager->appendPulse(np);
	}
	{
		OP_NumericParameter	np;

		np.name = "Cookeveryframe";
		np.label = "Cook Every frame";
		np.defaultValues[0] = 0;
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "Schroedingerintegration";
		np.label = "Schroedinger Integration";
		np.defaultValues[0] = 0;
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "N";
		np.label = "N";
		np.defaultValues[0] = 1;
		np.defaultValues[1] = 1;
		np.defaultValues[2] = 1;
		OP_ParAppendResult res = manager->appendInt(np, 3);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "Threadsperblock";
		np.label = "Threads per block";
		np.defaultValues[0] = 8;
		np.defaultValues[1] = 8;
		np.defaultValues[2] = 8;
		OP_ParAppendResult res = manager->appendInt(np, 3);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "Hbar";
		np.label = "Hbar";
		np.defaultValues[0] = 0.1;
		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "Dt";
		np.label = "Dt";
		np.defaultValues[0] = 0.0416666667;
		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "L";
		np.label = "L";
		np.defaultValues[0] = 1.;
		np.defaultValues[1] = 1.;
		np.defaultValues[2] = 1.;
		OP_ParAppendResult res = manager->appendFloat(np, 3);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "Prescribedvelocity";
		np.label = "Prescribed velocity";
		np.defaultValues[0] = 0;
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;

		np.name = "Loadprescribedvelocityeachframe";
		np.label = "Load prescribed velocity each frame";
		np.defaultValues[0] = 0;
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;
		np.name = "Repeatpressureprojection";
		np.label = "Repeat pressure projection";
		np.defaultValues[0] = 1;
		OP_ParAppendResult res = manager->appendInt(np,1);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;
		np.name = "Applyexternalforce";
		np.label = "Apply external force";
		np.defaultValues[0] = 0;
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
	{
		OP_NumericParameter	np;
		np.name = "Externalforce";
		np.label = "External force";
		np.defaultValues[0] = 0;
		np.defaultValues[1] = 0;
		np.defaultValues[2] = 0;

		OP_ParAppendResult res = manager->appendFloat(np, 3);
		assert(res == OP_ParAppendResult::Success);
	}
}

void
ISFTOP::pulsePressed(const char* name, void* reserved)
{
	if (!strcmp(name, "Reset"))
	{
		reset = true;
	}
}
