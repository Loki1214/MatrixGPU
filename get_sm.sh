#!/bin/sh
source=$(mktemp).cu
target=get_sm

trap "rm -f $source ./$target" 0 1 2 3 15

cat << EOF > $source
#include <cassert>
#define cuCHECK(call)                                                          \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if(error != cudaSuccess) {                                             \
            printf("cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            assert(error == cudaSuccess);                                      \
        }                                                                      \
    };
#include <vector>
#include <algorithm>
#include <iostream>
int main() {
	int ngpus = 0;
	cuCHECK(cudaGetDeviceCount(&ngpus));
    std::vector<int> sm(ngpus,0);
    for(int i = 0;i < ngpus; ++i) {
        cudaDeviceProp prop;
        cuCHECK(cudaGetDeviceProperties(&prop, i));
        sm[i] = prop.major * 10 + prop.minor;
    }
    std::sort(sm.begin(), sm.end());
    int size = std::unique(sm.begin(), sm.end()) - sm.begin();

    std::cout << sm[0];
    for(int j = 1;j < size; ++j) std::cout << ";" << sm[j];
    std::cout << std::endl;
}
EOF

nvcc -o $target $source && ./$target