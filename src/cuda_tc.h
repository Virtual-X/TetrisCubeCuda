/*
 * cuda_tc.h
 *
 *  Created on: Mar 4, 2014
 *      Author: igor
 */

#ifndef CUDA_TC_H_
#define CUDA_TC_H_

#if defined(_MSC_VER)
    //  Microsoft
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#else
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#endif

#ifdef DLL_EXPORTS
#define API EXPORT
#else
#define API IMPORT
#endif

#include <stdint.h>

extern "C" {

#define TETRIS_4
#ifdef TETRIS_4
enum {
	piecesCount = 12,
	gridSize = 64
};
#else
enum {
	piecesCount = 6,
	gridSize = 27
};
#endif

struct solverStatus {
	uint64_t grid;
	int actualPiece;
	int position[piecesCount];
	int currentCandidatesIndex[piecesCount];
	int permutatorIndices[piecesCount];
	int permutatorObjects[piecesCount];
};

struct solution {
	int candidateIndex[piecesCount];
	//solution* next;
};

#define __host_solver__

#ifdef __host_solver__
int API SolveCPU(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* solutions,
		int minPiece,
		int maxSolutions);
#endif

int API SolveGPU(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* solutions,
		int minPiece,
		int maxSolutions,
		int solversCount);

int API SplitCPU(
		int splitLevel,
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* status);

} // extern "C"

#endif /* CUDA_TC_H_ */
