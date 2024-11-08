#include <linux/perf_event.h> /* Definition of PERF_* constants */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>
#include <inttypes.h>
#include <functional>

// CPU has only 4 counter registers available for developer in linux
//
#define TOTAL_EVENTS 4

enum COUNTERS{
  NONE   = 0,
  INSTR  = 1,
  BRANCH = 2,
  CACHE  = 3,
  TLB    = 4
}; 

// Executes perf_event_open syscall and makes sure it is successful or exit
long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);

// Helper function to setup a perf event structure (perf_event_attr; see man perf_open_event)
void configure_event(struct perf_event_attr *pe, uint32_t type, uint64_t config);

void invalid_profile_mode();

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

COUNTERS configure_counters(perf_event_attr* pe);

void print_results(uint64_t *pe_val, COUNTERS mode);

template <class Function, typename... Args>
auto run_with_counters(Function &&F, Args &&...ArgList)
{
  bool enable_profiling = false;
  // fd[0] will be the group leader file descriptor
  // 
  int      fd[TOTAL_EVENTS];
  // event ids for file descriptors
  //
  int      id[TOTAL_EVENTS];
  // Counter value array corresponding to fd/id array
  //
  uint64_t pe_val[TOTAL_EVENTS];
  // Configuration structure for perf events (see man perf_event_open)
  //
  struct   perf_event_attr pe[TOTAL_EVENTS];
  struct   read_format counter_results;

  COUNTERS mode = configure_counters(pe);
  
  if (mode != COUNTERS::NONE) {
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

  std::invoke(F,ArgList...);

  if (mode != COUNTERS::NONE) {
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

  print_results(pe_val, mode);

  // Close counter file descriptors
  for(int i = 0; i < TOTAL_EVENTS; i++){
    close(fd[i]);
  }
}
