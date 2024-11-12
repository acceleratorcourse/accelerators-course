                                             Learn to use performance analyzing tools for cpu in Linux

*    Install gnu perf utility
```
    apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
```

*    Compile simple program
```
    cd lab1/daxpy
    g++ daxpy.cpp -O3 -std=c++17 -o daxpy
```

*    Try to use gnu perf utility on compiled program
```
    perf stat ./daxpy 100000000 10 1
```

*    Repeat 10 times to observe variance
```
    for i in seq{1..10};do perf stat ./daxpy 100000000 10 1;done
```

*    Check cache misses
```
    for i in seq{1..10};do perf stat  -e cache-references,cache-misses,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores ./daxpy 1000000 10 1; done
```

*    Check TLB statistics
```
    for i in seq{1..10};do perf stat  -e TLB-loads,dTLB-load-misses,dTLB-prefetch-misses ./daxpy 1000000 10 1; done
```

*    Check L2 caches
```
    for i in seq{1..10};do perf stat  -e LLC-loads,LLC-load-misses,LLC-stores,LLC-prefetches ./daxpy 1000000 10 1; done
```

*    Trace counters with small time intervals
```
    perf stat -I 100 ./daxpy 10000000 100 1
```

*    Try to run with different workload and observe variance
*    Run with different external and internal loop induction variable upper bounds and strides. Observe differences, make conclusions
```
      ./daxpy 1000000000 1 1
      ./daxpy 100000000 10 1
      ./daxpy 10000000 100 1
      ./daxpy 1000000 1000 1
      ./daxpy 100000 10000 1
      ./daxpy 10000 100000 1
      for i in {1..10};do ./daxpy 10000000 100 $i;done
```

*    Change data type to float or char and run all tests again
  
*    Compile example with custom profiling function
```
     g++ function_profiling_example.cpp perf_count.cpp -O3 -std=c++17 -o profiled_daxpy
     PROFILE_INSTRUCTIONS=1 ./profiled_daxpy 1000000 10 1
     PROFILE_BRANCHES=1 ./profiled_daxpy 1000000 10 1
     PROFILE_L1_CACHES=1 ./profiled_daxpy 1000000 10 1
     PROFILE_TLB=1 ./profiled_daxpy 1000000 10 1
```  
*    Try to break branch prediction

Homework: 
1. Try to write simple benchmark that will allow you to estimate L1 cacheline size of your CPU using gnu perf utility
2. Try to write simple benchmark will allow you to estimate L1 cache size and it's associativity 
3. Try to write simple benchmark that will allow you to estimate L2 and LLC cache sizes and their associativity
4. Try to write simple benchmark that will allow you to estimate TLB cache size and system virtual page size
5. Try to write simple benchmark that will allow you to demonstrare work of branch predictor

Suggested literature Computer Architecture: A Quantitative Approach
Book by David A Patterson and John L. Hennessy


  


