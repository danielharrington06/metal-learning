#include <metal_stdlib>
using namespace metal;

// partial product stage of dot product
kernel void multiply_vectors(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * b[index];
}

// reduce sum
kernel void reduce_sum(
    device float* values [[buffer(0)]],
    device float* partialSums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& threadgroupSize [[buffer(3)]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint groupIndex [[thread_position_in_grid]]
) {
    threadgroup float sharedData[1024];

    uint globalIndex = groupIndex * threadgroupSize + threadIndex;

    if (globalIndex < count) {
        sharedData[threadIndex] = values[globalIndex];
    } else {
        sharedData[threadIndex] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup); // doesnt let any thread continue until the relevant threads have reached this point, so prevents race condition

    for (uint stride = threadgroupSize/2; stride > 0; stride /= 2) {
        if (threadIndex < stride) {
            sharedData[threadIndex] += sharedData[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        partialSums[groupIndex] = sharedData[0];
    }

    // reduction implementation will go here
}