#include <cstdlib>
#include <inttypes.h>
#include <stdio.h>
#include <vector>

/*
template <typename T>
void daxpy(size_t n, size_t stride, T a, T *x, T *y) {
    for(size_t i = 0; i < n; i+=stride) {
        y[i] = a * x[i] + y[i];
    }
}*/

template <typename T>
void daxpy(size_t n, size_t stride, T a, T *x, T *y) {
    int res = 0;
    for(size_t i = 0; i < n; i+=stride) {
       res += a * x[i];
    }
    x[0] = res;
}


template <typename T> 
void run_daxpy(size_t n, size_t stride, size_t repeats) {
   
   std::vector<T> x,y;
   //size_t total_size = stride * n;
   size_t total_size = n;
   x.reserve(total_size);
   y.reserve(total_size);
   printf("input size in bytes : %" PRIu64 "\n", x.capacity()*sizeof(T));

   T a = 3;
   for (int i = 0; i < repeats; i++) {
       daxpy(n, stride, a, x.data(), y.data());
   }
}

/*
 *   gcc with -O3 optimization
 *   inlines function
 *   and unrolls both loops
 *
 *   for (j = 0; j < repeats/2; j+=2) { 
 *     for(size_t i = 0; i < n/2; i+=stride*2) {
 *       y[i]   = a * x[i]   + y[i];
 *       y[i]   = a * x[i]   + y[i];
 *       y[i+1] = a * x[i+1] + y[i+1];
 *       y[i+1] = a * x[i+1] + y[i+1];
 *     }
 *   }
 *
 *   process tail of loop...
 *
 *   compile with -O2 flag to prevent unrolling
*/

int main(int argc, char** argv) {
  size_t n       = std::atoll(argv[1]);
  size_t repeats = std::atoll(argv[2]);
  size_t stride  = std::atoll(argv[3]);

  printf("n : %" PRIu64 "\n", n);
  printf("repeats : %" PRIu64 "\n", repeats);
  printf("stride  : %" PRIu64 "\n", stride);

  run_daxpy<char>(n, stride, repeats);

  return 0;
}

