                                             Learn to use performance analyzing tools for cpu in Linux

*    Install gnu perf utility
```
    apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
```
*    compile simple program
```
    cd lab1/daxpy
    g++ daxpy.cpp -o daxpy
```
*    try to use gnu perf utility on compiled program
```
    perf stat ./daxpy 100000000 10 1
```
*    repeat 10 times to observe variance
```
    for i in seq{1..10};do perf stat ./daxpy 100000000 10 1;done
```
*    check cache misses
```
    for i in seq{1..10};do perf stat  -e cache-references,cache-misses,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores ./daxpy 1000000 10 1; done
```
*    check TLB statistics
```
    for i in seq{1..10};do perf stat  -e TLB-loads,dTLB-load-misses,dTLB-prefetch-misses ./daxpy 1000000 10 1; done
```
*    check L2 caches
```
    for i in seq{1..10};do perf stat  -e LLC-loads,LLC-load-misses,LLC-stores,LLC-prefetches /daxpy 1000000 10 1; done
```
*    trace counters with small time intervals
```
    perf stat -I 100 ./daxpy 10000000 100 1
```
*    try to run with different workload and observe variance
*    run with different external and internal loop induction variable upper bounds and observe differences, make conclusions
```
      ./daxpy 1000000000 1 1
      ./daxpy 100000000 10 1
      ./daxpy 10000000 100 1
      ./daxpy 1000000 1000 1
      ./daxpy 100000 10000 1
      ./daxpy 10000 100000 1
```
  


