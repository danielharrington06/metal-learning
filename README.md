# Metal Learning

I am using this project to learn how to work with the GPU in order to parallelise computations, where possible.

# Understanding So Far

`vector_add.metal` contains the code that says for each thread, what to do and where to put the result. Here, it says to take one element from `a` and `b`, add them, and put the result in `result`.



# Compile & Run

```bash
xcrun -sdk macosx metal -c vector_add.metal -o vector_add.air
xcrun -sdk macosx metallib vector_add.air -o vector_add.metallib
```