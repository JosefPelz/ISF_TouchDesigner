# CudaSortTOP

TouchDesigner TOP wrapper for Cuda Thrust sort_by_key function.
Compatible with TouchDesigner 99 Spring 2019 Release and more recent.
Windows binary provided on the releases page https://github.com/vinz9/CudaSortTOP/releases

## Usage

CudaSortTOP.dll should go in Derivative plugins folder, see https://docs.derivative.ca/Custom_Operators
or can be loaded with the cplusplusTOP for debugging.
A sample Toe is provided.
Input Texture needs to be 16-bit float (RGBA) or 32-bit float (RGBA).
The Cuda Sort TOP output format needs to be 32-bit float (Mono).
Sorts indices using a key. Points needs to be reordered using the sorted indices.

## Compilation

Compiled with Visual Studio 2015 and Cuda 9.2

## Disclaimer
This is provided as is, mainly as a starting point for people interested in extending TouchDesigner with Cuda.

Vincent Houzé
https://vincenthouze.com