# Metal Learning

I am using this project to learn how to work with the GPU in order to parallelise computations, where possible.

## Understanding So Far

### 1. `vector_add`

- `vector_add.metal` contains the code that says for each thread, what to do and where to put the result. Here, it says to take one element from `a` and `b`, add them, and put the result in `result`.

### 2. `vector_transform`
- This code similarly makes use of GPU kernel for parallelised computation, this time just a single input to single output.
- A CPU version of the transformation, `transformCPU`, was added to confirm correctness.

### 3. `dot_products`
- This code calculates the dot_product of two vectors on the GPU by parallelising the multiplication, then using a tree-reduction approach to parallelise the summation.

## Benchmarking

I carried out benchmarking for calculating the dot product of two vectors as this involved parallel computations on respective elements on respective elements of a vector then parallel reduction, making it more interesting than the first two computations that I wrote.

I used 50 warmup runs and 200 test runs.

```bash
Dot Product Benchmark
Vectors of size 100000000
50 Warmup Runs
200 Test Runs

CPU Time Average: 264.437ms
GPU Time Average: 27.122ms
GPU speedup: 9.74991x
```

```bash
Dot Product Benchmark
Vectors of size 10000000
50 Warmup Runs
200 Test Runs

CPU Time Average: 26.019ms
GPU Time Average: 6.70938ms
GPU speedup: 3.87801x
```

## Understanding Metal Objects

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

## Compile & Run

Metal `.air` and `.metallib`
```bash
xcrun -sdk macosx metal -c shaders/filename.metal -o bin/filename.air
xcrun -sdk macosx metallib bin/filename.air -o bin/filename.metallib
```

Objective C
```bash
clang++ -std=c++17 src/filename.mm \
    -framework Metal \
    -framework Foundation \
    -o bin/filename
```
