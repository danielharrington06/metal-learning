# Metal Learning

I am using this project to learn how to work with the GPU in order to parallelise computations, where possible.

# Understanding So Far

`vector_add.metal` contains the code that says for each thread, what to do and where to put the result. Here, it says to take one element from `a` and `b`, add them, and put the result in `result`.

# Understanding Metal Objects

```bash
MTLDevice
    -> creates
MTLCommandQueue
    -> creates
MTLCommandBuffer
    -> creates
MTLComputeCommandEncoder
    -> encodes
GPU kernel
```

```bash
CPU memory
    -> copy
MTLBuffer
    -> GPU reads/writes
Apple GPU
```

- MTLDevice
    - represents the GPU
- MTLCommandQueue
    - a queue where we submit work for the GPU
- MTLCommandBuffer
    - contains a batch of commands that will be executed by the GPU
- MTLComputerCommandEncoder
    - used to tell Metal which computer kernel to execute with which buffers with how many threads
- MTLBuffer
    - a region of memory accessible to the GPU

# Compile & Run

```bash
xcrun -sdk macosx metal -c vector_add.metal -o vector_add.air
xcrun -sdk macosx metallib vector_add.air -o vector_add.metallib
```