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

