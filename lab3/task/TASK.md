# Lab2

##Task 1

Replace body of function RunGPUImplNaive method with your version of matrix multimplication

##Task 2

Read [optimization hints](https://accelerated-computing-class.github.io/fall24/labs/lab4/) part 2
and implement your version of RunGPUImplLDS function using local data storage memory for caching

##Task 3

Read [optimization hints](https://accelerated-computing-class.github.io/fall24/labs/lab4/) part 3
and implement your version of RunGPUImplRegisters function using registers for caching
Check the code emitted by compiler(ptx for Nvidia/assembly for Amd)

##Task 4

Read [optimization hints](https://accelerated-computing-class.github.io/fall24/labs/lab6/)
and implement your version of RunGPUImplTensorCore function using wmma instructions
Check the code emitted by compiler(ptx for Nvidia/assembly for Amd)

##Task 5

Profile CPU implementation of cblas_sgemm using methods described in lab1
Profile GPU implementation of rocblas_sgemm/cublas_sgemm using 
[gpu profiler](https://developer.nvidia.com/nsight-systems)
Run on different input sizes, analyze performance and try to make conclusions 

