#include <cstdlib>
#include <inttypes.h>

void daxpy(size_t n, size_t stride, double a, double*x, double*y) {
    for(size_t i = 0; i < n; i+=stride) {
        y[i] = a * x[i] + y[i];
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
  printf("stride : %" PRIu64 "\n", stride);

  double* x = (double*)malloc(n * sizeof(double));
  double* y = (double*)malloc(n * sizeof(double));
  double  a = 1.5;

  for(int i = 0; i < repeats; i++) {
      daxpy(n,stride, a, x, y);
  }

  // prevent removing daxpy by compiler
  //
  printf("%f", x[0]);

  free(x);
  free(y);
    
  return 0;
}

