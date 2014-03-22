/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#define DLL_EXPORTS

#include "cuda_tc.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>

using thrust::device_vector;
using thrust::host_vector;
#define std_vector std::vector

template<typename T>
static T* raw(std_vector<T>& vector)
{
	return &vector[0];
}

#ifdef __host_solver__
#define __solver_access__ __host__ __device__
#else
#define __solver_access__ __device__
#endif

static const char* filePath = "/home/igor/Development/cuda-workspace/cuda_tetriscube/data/solve_call";

bool fileExist(const std::string& name) {
	std::ifstream file(name.c_str());
	return file;
}

std::string nextFile() {
	std::ostringstream s;
	int n = 0;
	do {
		s.clear();
		s.str("");
		s << filePath << n;
		n++;
	} while (fileExist(s.str()));
	return s.str();
}

std::string lastFile() {
	std::ostringstream s;
	std::string last;
	int n = 0;
	do {
		last = s.str();
		s.clear();
		s.str("");
		s << filePath << n;
		n++;
	} while (fileExist(s.str()));
	return last;
}

void write(std::ofstream& file, const void* value, size_t size) {
    file.write(static_cast<const char*>(value), size);
}

void read(std::ifstream& file, void* value, size_t size) {
    file.read(static_cast<char*>(value), size);
}

void SaveToFile(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		int minPiece,
		int maxSolutions,
		int solversCount)
{
//		piecesCount = 12,
//		gridSize = 64,
//		solutionBufferSize = 5
	// candidates[candidateOffsets[gridSize * piecesCount]]
	// candidateOffsets[gridSize * piecesCount + 1] (gridIndex * piecesCount + pieceIndex)
	// solversStatus[solversCount]

    std::string name = nextFile();
    std::ofstream file(name.c_str(), std::ios::binary);
    char v0[] = { (char)sizeof(int), (char)sizeof(uint64_t) };
    write(file, v0, sizeof(v0));
    int v1[] = { sizeof(solverStatus), piecesCount, gridSize, 0 };
    write(file, v1, sizeof(v1));
    write(file, candidateOffsets, (gridSize * piecesCount + 1) * sizeof(int));
    write(file, candidates, candidateOffsets[gridSize * piecesCount] * sizeof(uint64_t));
    int v2[] = { minPiece, maxSolutions, solversCount };
    write(file, v2, sizeof(v2));
    write(file, solversStatus, solversCount * sizeof(solverStatus));
}

static void TestEqual(int actual, int expected, const std::string& varname) {
	if (actual != expected) {
		std::ostringstream s;
		s << "Invalid " << varname << " (actual " << actual << ", expected " << expected << ")";
		throw s.str();
	}
}

void LoadFromFile(
		std_vector<uint64_t>& candidates,
		std_vector<int>& candidateOffsets,
		std_vector<solverStatus>& solversStatus,
		int& minPiece,
		int& maxSolutions,
		int& solversCount)
{
//		piecesCount = 12,
//		gridSize = 64,
//		solutionBufferSize = 5
	// candidates[candidateOffsets[gridSize * piecesCount]]
	// candidateOffsets[gridSize * piecesCount + 1] (gridIndex * piecesCount + pieceIndex)
	// solversStatus[solversCount]

	std::string name = lastFile();
	if (name.empty()) {
		throw "No last file";
	}
	std::ifstream file(name.c_str(), std::ios::binary);
    char v0[2];
    read(file, v0, sizeof(v0));
    TestEqual(v0[0], sizeof(int), "int size");
    TestEqual(v0[1], sizeof(uint64_t), "uint64_t size");
    int v1[4];
    read(file, v1, sizeof(v1));
    TestEqual(v1[0], sizeof(solverStatus), "solverStatus size");
    TestEqual(v1[1], piecesCount, "piecesCount");
    TestEqual(v1[2], gridSize, "gridSize");
    // dummy v1[3]
	candidateOffsets.resize(gridSize * piecesCount + 1);
    read(file, raw(candidateOffsets), candidateOffsets.size() * sizeof(int));
	candidates.resize(candidateOffsets[gridSize * piecesCount]);
    read(file, raw(candidates), candidates.size() * sizeof(uint64_t));
    int v2[3];
    read(file, v2, sizeof(v2));
    minPiece = v2[0];
    maxSolutions = v2[1];
    solversCount = v2[2];
	solversStatus.resize(solversCount);
    read(file, raw(solversStatus), solversStatus.size() * sizeof(solverStatus));
}

class solverStata {
public:
	solverStata(solverStatus* status, int solversCount) {
		grid = getMember(status, solversCount, &solverStatus::grid);
		actualPiece = getMember(status, solversCount, &solverStatus::actualPiece);
		for (int i = 0; i < piecesCount; i++) {
			position[i] = getMember(status, solversCount, &solverStatus::position, i);
			currentCandidatesIndex[i] = getMember(status, solversCount, &solverStatus::currentCandidatesIndex, i);
			permutatorIndices[i] = getMember(status, solversCount, &solverStatus::permutatorIndices, i);
			permutatorObjects[i] = getMember(status, solversCount, &solverStatus::permutatorObjects, i);
		}
	}

private:

	template<typename T>
	static host_vector<T> getMember(solverStatus* status, int solversCount, T solverStatus::*member) {
		host_vector<T> result(solversCount);
		for (int i = 0; i < solversCount; i++)
			result[i] = status[i].*member;
		return result;
	}

	template<typename T>
	static host_vector<T> getMember(solverStatus* status, int solversCount, T (solverStatus::*member)[piecesCount], int index) {
		host_vector<T> result(solversCount);
		for (int i = 0; i < solversCount; i++)
			result[i] = (status[i].*member)[index];
		return result;
	}

	device_vector<uint64_t> grid;
	device_vector<int> actualPiece;
	device_vector<int> position[piecesCount];
	device_vector<int> currentCandidatesIndex[piecesCount];
	device_vector<int> permutatorIndices[piecesCount];
	device_vector<int> permutatorObjects[piecesCount];
};


enum {
	solutionsBufferSize = 10000
};

class solver
{
public:
	__solver_access__
	solver(const uint64_t* candidates,
			const int* candidateOffsets,
			solverStatus& status,
			solution* solutions,
			int* solutionsCount)
: candidates(candidates),
  candidateOffsets(candidateOffsets),
  status(status),
  solutions(solutions),
  solutionsCount(solutionsCount) {
}

	__solver_access__
	void DoStep();

#ifdef __host_solver__
	__host__
	void Split(int level,
			const uint64_t* candidates,
			const int* candidateOffsets,
            solverStatus* statusBuffer,
			int& n) const {
        solverStatus tempStatus = status;
		solver temp(candidates, candidateOffsets, tempStatus, 0, 0);
		while (temp.Next()) {
			solverStatus ss = tempStatus;
			solver s(candidates, candidateOffsets, ss, 0, 0);
			s.IncreaseActualPiece();
			if (level <= 1) {
				if (statusBuffer)
					statusBuffer[n] = ss;
				n++;
			}
			else {
				s.Split(level - 1, candidates, candidateOffsets, statusBuffer, n);
			}
		}
	}
#endif

private:

	__solver_access__
	static bool IsValid(uint64_t candidate, uint64_t grid);

	__solver_access__
	void Swap(int i, int j);

	__solver_access__
	bool Next();

	__solver_access__
	void DecreaseActualPiece();

	__solver_access__
	void IncreaseActualPiece();

	__solver_access__
	void AddSolution();

	const uint64_t* candidates;
	const int* candidateOffsets;
	solverStatus& status;
	solution* solutions;
	int* solutionsCount;
};

__solver_access__
bool solver::IsValid(uint64_t candidate, uint64_t grid)
{
    return (candidate & grid) == 0;
}

__solver_access__
void solver::Swap(int i, int j)
{
	if (i == j)
		return;
	int* objects = status.permutatorObjects;
    int t = objects[i];
    objects[i] = objects[j];
    objects[j] = t;
}

__solver_access__
bool solver::Next()
{
	//const int& actualPiece = status.actualPiece;
	int actualPiece = status.actualPiece;
	int candidatesOffset = status.position[actualPiece] * piecesCount;

    //int index = status.currentCandidatesIndex[actualPiece];
    int& index = status.currentCandidatesIndex[actualPiece];
    index++;

	int candidateNumber = actualPiece + status.permutatorIndices[actualPiece];
    Swap(actualPiece, candidateNumber);
	for (; candidateNumber < piecesCount; candidateNumber++) {
    	int candidatesIndex = candidatesOffset + status.permutatorObjects[candidateNumber];
    	int min = candidateOffsets[candidatesIndex];
    	int max = candidateOffsets[candidatesIndex + 1];
        if (index < min)
            index = min;
        while (index < max) {
            if (IsValid(candidates[index], status.grid)) {
            	//status.currentCandidatesIndex[actualPiece] = index;
            	status.permutatorIndices[actualPiece] = candidateNumber - actualPiece;
                Swap(actualPiece, candidateNumber);
                return true;
            }
            index++;
        }

        index = -1;
    }
	status.permutatorIndices[actualPiece] = 0;
    //status.currentCandidatesIndex[actualPiece] = -1;
    return false;
}

__solver_access__
void solver::DecreaseActualPiece() {
    int& actualPiece = status.actualPiece;

    actualPiece--;

    if (actualPiece >= 0) {
    	int pieceIndex = status.currentCandidatesIndex[actualPiece];
    	status.grid &= ~candidates[pieceIndex];
    }
}

__solver_access__
void solver::IncreaseActualPiece() {
    int& actualPiece = status.actualPiece;

	int pieceIndex = status.currentCandidatesIndex[actualPiece];
	status.grid |= candidates[pieceIndex];

    actualPiece++;

    if (actualPiece < piecesCount) {
        int pos = status.position[actualPiece - 1] + 1;
        while ((status.grid & (1ULL << pos)) > 0)
        	pos++;
        status.position[actualPiece] = pos;
#ifndef __CUDA_ARCH__
//        static int x = 0;
//        if (x++ < 15)
//        	std::cout << "pos " << pos << std::endl;
#endif
    }
}

__solver_access__
void solver::AddSolution() {
#ifdef __CUDA_ARCH__
	int index = atomicAdd(solutionsCount, 1);
#else
	int index = *solutionsCount;
	++(*solutionsCount);
#endif
	if (index >= solutionsBufferSize)
		return;
	const int* current = status.currentCandidatesIndex;
	int* solution = solutions[index].candidateIndex;
	for (int i = 0; i < piecesCount; i++) {
		solution[i] = current[i];
	}
}

__solver_access__
void solver::DoStep() {
	if (!Next()) {
		DecreaseActualPiece();
	} else {
		IncreaseActualPiece();
		if (status.actualPiece == piecesCount) {
			AddSolution();
			DecreaseActualPiece();
		}
	}
}

// todo: try kernel steps
// todo: try local data instead of shared (copy all)

__solver_access__
bool SolveSingle(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus& status,
		solution* solutions,
		int* solutionsCount,
		int minPiece,
		int maxSteps)
{
	solver solver(candidates, candidateOffsets, status, solutions, solutionsCount);

	int step = 0;
	while (step++ < maxSteps) {
		solver.DoStep();
		if (status.actualPiece < minPiece) {
			return true;
		}
	}

	return false;
}

__global__
void SolveKernel(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* solutions,
		int* solutionsCount,
		int minPiece,
		int solversCount,
		int maxSteps)
{
	// candidates[candidateOffsets[gridSize * piecesCount]]
	// candidateOffsets[gridSize * piecesCount + 1] (gridIndex * piecesCount + pieceIndex)
	// solversStatus[solversCount]
	// solutions[solversCount]
	const int solverIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (solverIndex >= solversCount)
		return;

	solverStatus status(solversStatus[solverIndex]);
	SolveSingle(
			candidates,
			candidateOffsets,
			status,
			solutions,
			solutionsCount,
			minPiece,
			maxSteps);
	solversStatus[solverIndex] = status;
}

__global__
void RemoveFinishedKernelA(
		solverStatus* solversStatus,
		int minPiece,
		int solversCount,
		int* newSolversCount,
		int* solverToGrab)
{
	const int solverIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (solverIndex >= solversCount)
		return;

	bool finished = solversStatus[solverIndex].actualPiece < minPiece;
	solverToGrab[solverIndex] = finished ? (atomicSub(newSolversCount, 1) - 1) : -1;
}

__global__
void RemoveFinishedKernelB(
		solverStatus* solversStatus,
		int minPiece,
		int solversCount,
		int* solverToGrab)
{
	const int solverIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (solverIndex >= solversCount) // must be the reduced count!
		return;

	int grab = solverToGrab[solverIndex];
	if (grab < 0)
		return;

	while (solverToGrab[grab] >= 0)
		grab = solverToGrab[grab];

	solversStatus[solverIndex] = solversStatus[grab];
}

template<typename T>
__host__
static T* raw(device_vector<T>& vector)
{
	return thrust::raw_pointer_cast(vector.data());
}

int SolveGPU_T(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* preallocatedSolutions,
		std::list<solution>* solutionsList,
		int minPiece,
		int maxSolutions,
		int solversCount)
{
//	solverStata sx(solversStatus, solversCount);

	const int candidatesCount = gridSize * piecesCount;
	device_vector<uint64_t> dCandidates(candidates, candidates + candidateOffsets[candidatesCount]);
	device_vector<int> dCandidateOffsets(candidateOffsets, candidateOffsets + candidatesCount);
	device_vector<solverStatus> dSolversStatus(solversStatus, solversStatus + solversCount);

	device_vector<solution> dSolutions(solutionsBufferSize);
	device_vector<int> dSolutionsCount(1);

	device_vector<int> dSolversCount(1);
	dSolversCount[0] = solversCount;

	device_vector<int> dSolverToGrab(solversCount);

	std::cout << " pass: " << -1 << ", solvers " << solversCount <<  std::endl;

	const uint64_t* csr = raw(dCandidates);
	const int* cosr = raw(dCandidateOffsets);
	solverStatus* sssr = raw(dSolversStatus);
	solution* ssr = raw(dSolutions);
	int* ssc = raw(dSolutionsCount);
	int* dsc = raw(dSolversCount);
	int* dstg = raw(dSolverToGrab);

	const int blockSize = 512;

	cudaEvent_t startT;
	cudaEventCreate(&startT);

	cudaEventRecord(startT, 0);

	do {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		for (int i = 0; i < 5; i++) {
			const int steps = 1000 + (10 * solversCount / dSolversCount[0]);

			SolveKernel<<<(dSolversCount[0]+blockSize-1)/blockSize, blockSize>>>(csr, cosr, sssr, ssr, ssc, minPiece, dSolversCount[0], steps);
			RemoveFinishedKernelA<<<(dSolversCount[0]+blockSize-1)/blockSize, blockSize>>>(sssr, minPiece, dSolversCount[0], dsc, dstg);
			RemoveFinishedKernelB<<<(dSolversCount[0]+blockSize-1)/blockSize, blockSize>>>(sssr, minPiece,  dSolversCount[0], dstg);

//			std::cout << " pass:";
			std::cout << " " << i << std::flush;
//			std::cout << ", solvers " << dSolversCount[0];
//			std::cout << std::endl;
			if (dSolversCount[0] == 0)
				break;
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);

		std::cout << std::endl << "Cycle time: " << elapsedTime / 1000 << " s" << std::endl;

		cudaEventElapsedTime(&elapsedTime, startT, stop);

		std::cout << "Solve time: " << elapsedTime / 1000 << " s" << std::endl;

		int solutionsC = dSolutionsCount[0];
		std::cout << "Solutions: " << solutionsC << ", solvers finished "
				<< (solversCount - dSolversCount[0]) << "/" << solversCount << std::endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	while (dSolversCount[0] > 300000);

	cudaEventDestroy(startT);

	if (preallocatedSolutions) {
		int count = std::min(maxSolutions, (int)dSolutionsCount[0]);
		thrust::copy(dSolutions.begin(), dSolutions.begin() + count, preallocatedSolutions);
	}
	if (solutionsList) {
		host_vector<solution> sols(dSolutions.begin(), dSolutions.begin() + dSolutionsCount[0]);
		solutionsList->insert(solutionsList->end(), sols.begin(), sols.end());
	}

	return (int)dSolutionsCount[0];
}

int SolveGPU_(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* preallocatedSolutions,
		std::list<solution>* solutionsList,
		int minPiece,
		int maxSolutions,
		int solversCount)
{
#ifndef __APP__
	SaveToFile(candidates, candidateOffsets, solversStatus, minPiece, maxSolutions, solversCount);
#endif

	return SolveGPU_T(
			candidates,
			candidateOffsets,
			solversStatus,
			preallocatedSolutions,
			solutionsList,
			minPiece,
			maxSolutions,
			solversCount);
}

int SolveGPU(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* solutions,
		int minPiece,
		int maxSolutions,
		int solversCount)
{
#ifndef __APP__
	SaveToFile(candidates, candidateOffsets, solversStatus, minPiece, maxSolutions, solversCount);
#endif

	return SolveGPU_(candidates, candidateOffsets, solversStatus, solutions, 0, minPiece, maxSolutions, solversCount);
}

#ifdef __host_solver__

int SolveCPU(
		const uint64_t* candidates,
		const int* candidateOffsets,
		solverStatus* solversStatus,
		solution* solutions,
		int minPiece,
		int maxSolutions)
{
	int count = 0;
	while (SolveSingle(
				candidates,
				candidateOffsets,
				*solversStatus,
				solutions,
				&count,
				minPiece,
				100000));
	return count;
}

#endif

static void Init(solverStatus& status) {
    status.actualPiece = 0;
    status.grid = 0;
    status.position[0] = 0;
    for (int i = 0; i < piecesCount; i++) {
        status.currentCandidatesIndex[i] = -1;
        status.permutatorIndices[i] = 0;
        status.permutatorObjects[i] = i;
        status.position[i] = 0;
    }
}

int SplitCPU(int splitLevel,
             const uint64_t* candidates,
             const int* candidateOffsets,
             solverStatus* status) {
    if (splitLevel < 1) {
        if (status)
            Init(*status);
        return 1;
    }

    int n = 0;
    solverStatus ss;
    Init(ss);
    solver s(candidates, candidateOffsets, ss, 0, 0);
    s.Split(splitLevel, candidates, candidateOffsets, status, n);
    return n;
}

#ifdef __APP__
int main()
{
	std::cout << "main" << std::endl;
	std_vector<uint64_t> candidates;
	std_vector<int> candidateOffsets;
	std_vector<solverStatus> solversStatus;
	int minPiece;
    int maxSolutions;
	int solversCount;

	try {
		LoadFromFile(candidates, candidateOffsets, solversStatus, minPiece, maxSolutions, solversCount);
	} catch (std::string& error) {
		std::cout << error  << std::endl;
		return 1;
	}

	std::list<solution> solutionsList;

	SolveGPU_(
			raw(candidates),
			raw(candidateOffsets),
			raw(solversStatus),
			0,
			&solutionsList,
			minPiece,
			maxSolutions,
			solversCount);

	std::cout << solutionsList.size() << std::endl;

	return 0;
}
#endif
