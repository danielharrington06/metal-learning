#include <metal_stdlib>
using namespace metal;

// perform a transform on a vector
kernel void vector_transform(
    device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    float x = a[index];
    result[index] = sin(x * x + sqrt(abs(x)));
}