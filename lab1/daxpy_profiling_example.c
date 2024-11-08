#include <cstdlib>
#include <iostream>
#include <linux/perf_event.h> /* Definition of PERF_* constants */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>
#include <inttypes.h>

#define TOTAL_EVENTS 4

void daxpy(size_t n, size_t stride, double a, double*x, double*y) {
    for(size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Executes perf_event_open syscall and makes sure it is successful or exit
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags){
  int fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  if (fd == -1) {
    fprintf(stderr, "Error creating event");
    exit(EXIT_FAILURE);
  }

  return fd;
}

// Helper function to setup a perf event structure (perf_event_attr; see man perf_open_event)
void configure_event(struct perf_event_attr *pe, uint32_t type, uint64_t config){
  memset(pe, 0, sizeof(struct perf_event_attr));
  pe->type = type;
  pe->size = sizeof(struct perf_event_attr);
  pe->config = config;
  pe->read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  pe->disabled = 1;
  pe->exclude_kernel = 1;
  pe->exclude_hv = 1;
}

// Format of event data to read
// Note: This format changes depending on perf_event_attr.read_format
// See `man perf_event_open` to understand how this structure can be different depending on event config
// This read_format structure corresponds to when PERF_FORMAT_GROUP & PERF_FORMAT_ID are set
struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[TOTAL_EVENTS];
};

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

  auto measured_function = [&](){
    for(int i = 0; i < repeats; i++) {
      daxpy(n,stride, a, x, y);
    }
  };

  bool enable_profiling = false;
  int      fd[TOTAL_EVENTS];  // fd[0] will be the group leader file descriptor
  int      id[TOTAL_EVENTS];  // event ids for file descriptors
  uint64_t pe_val[TOTAL_EVENTS]; // Counter value array corresponding to fd/id array.
  struct   perf_event_attr pe[TOTAL_EVENTS];  // Configuration structure for perf events (see man perf_event_open)
  struct   read_format counter_results;

  if (getenv("PROFILE_INSTRUCTIONS")) {
    enable_profiling = true;
    // Configure the group of PMUs to count
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    configure_event(&pe[2], PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND);
    configure_event(&pe[3], PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
  }

  if (getenv("PROFILE_BRANCHES")) {
    enable_profiling = true;
    // Configure the group of PMUs to count
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_BPU);
    configure_event(&pe[2], PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    configure_event(&pe[3], PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
  }

  if (getenv("PROFILE_CACHES")) {
    enable_profiling = true;
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    configure_event(&pe[2], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_L1D) |
		                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
						(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    configure_event(&pe[3], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_L1D) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
  }

  if (getenv("PROFILE_TLB")) {
    enable_profiling = true;
    configure_event(&pe[0], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_DTLB) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    configure_event(&pe[1], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_DTLB) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
    configure_event(&pe[2], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_DTLB) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    configure_event(&pe[3], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_DTLB) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
  }

  if (enable_profiling) { 
    // Create event group leader
    fd[0] = perf_event_open(&pe[0], 0, -1, -1, 0);
    ioctl(fd[0], PERF_EVENT_IOC_ID, &id[0]);
    // Let's create the rest of the events while using fd[0] as the group leader
    for(int i = 1; i < TOTAL_EVENTS; i++){
      fd[i] = perf_event_open(&pe[i], 0, -1, fd[0], 0);
      ioctl(fd[i], PERF_EVENT_IOC_ID, &id[i]);
    }

    // Reset counters and start counting; Since fd[0] is leader, this resets and enables all counters
    // PERF_IOC_FLAG_GROUP required for the ioctl to act on the group of file descriptors
    ioctl(fd[0], PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
    ioctl(fd[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  }

  //function to profile
  measured_function();

  if (enable_profiling) {     
    // Stop all counters
    ioctl(fd[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

    // Read the group of counters and print result
    read(fd[0], &counter_results, sizeof(struct read_format));
    printf("Num events captured: %" PRIu64 "\n", counter_results.nr);
    for(int i = 0; i < counter_results.nr; i++) {
      for(int j = 0; j < TOTAL_EVENTS ;j++){
        if(counter_results.values[i].id == id[j]){
          pe_val[i] = counter_results.values[i].value;
        }
      }
    }
  }

  if (getenv("PROFILE_INSTRUCTIONS")) {
    printf("CPU cycles: %" PRIu64 "\n", pe_val[0]);
    printf("Instructions retired: %" PRIu64 "\n", pe_val[1]);
    printf("IPC :%f\n", pe_val[1] * 1.0/pe_val[0]);
    printf("Frontend stall cycles: %" PRIu64 "\n", pe_val[2]);
    printf("Backend stall cycles: %" PRIu64 "\n", pe_val[3]);
  }

  if (getenv("PROFILE_BRANCHES")) {
    printf("Instructions retired: %" PRIu64 "\n", pe_val[0]);
    printf("Branch prediction unit :%" PRIu64 "\n", pe_val[1]);
    printf("Branches: %" PRIu64 "\n", pe_val[2]);
    printf("Branch-misses: %" PRIu64 "\n", pe_val[3]);
    printf("Branch-miss rate :%f\n", pe_val[3] * 1.0/pe_val[2]);
  }

  if (getenv("PROFILE_CACHES")) {   
    printf("Cache references: %" PRIu64 "\n", pe_val[0]);
    printf("Cache misses :%" PRIu64 "\n", pe_val[1]);
    printf("L1 caches: %" PRIu64 "\n", pe_val[2]);
    printf("L1 cache misses: %" PRIu64 "\n", pe_val[3]);
    printf("L1 cache miss rate: %f\n", pe_val[3]*1.0/pe_val[3]);
  }

  if (getenv("PROFILE_TLB")) { 
    printf("TLB cache read: %" PRIu64 "\n", pe_val[0]);
    printf("TLB cache read misses :%" PRIu64 "\n", pe_val[1]);
    printf("TLB cache read miss rate :%f\n", pe_val[1]*1.0/pe_val[0]);
  }

    // Close counter file descriptors
  for(int i = 0; i < TOTAL_EVENTS; i++){
    close(fd[i]);
  }

  free(x);
  free(y);
    
  return 0;
}
