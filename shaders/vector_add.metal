#include <metal_stdlib>
using namespace metal;

// take one element from `a` and `b`, add them, and put the result in `result`.
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}