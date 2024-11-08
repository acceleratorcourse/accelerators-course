#include "perf_count.h"

long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags){
  int fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  if (fd == -1) {
    fprintf(stderr, "Error creating event");
    exit(EXIT_FAILURE);
  }
  return fd;
}

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

void invalid_profile_mode() {
  fprintf(stderr, "Profile mode is already set\n");
  exit(EXIT_FAILURE);
}

void print_instruction(uint64_t *pe_val) {
  printf("%-30s%15" PRIu64 "\n","CPU cycles:", pe_val[0]);
  printf("%-30s%15" PRIu64 "\n", "Instructions retired:",pe_val[1]);
  printf("%-30s%15f%s\n", "IPC:", pe_val[1] * 1.0/pe_val[0], "  insn per cycle");
  printf("%-30s%15" PRIu64 "\n", "Frontend stall cycles:", pe_val[2]);
  printf("%-30s%15.2f%%%s\n", "Frontend stall cycles rate:", pe_val[2]*100.0/pe_val[0], " frontend cycles idle");
  printf("%-30s%15" PRIu64 "\n", "Backend stall cycles:", pe_val[3]);
  printf("%-30s%15.2f%%%s\n", "Backend stall cycles rate:", pe_val[3]*100.0/pe_val[0], " backend cycles idle");
  printf("%-30s%15.2f%%%s\n", "Stall cycles rate:", (pe_val[3] + pe_val[2])*100.0/pe_val[0], " stalled cycles per insn");
}

void print_branch(uint64_t *pe_val) {
  printf("%-30s%15" PRIu64 "\n", "Instructions retired: ", pe_val[0]);
  printf("%-30s%15" PRIu64 "\n", "Branch prediction unit:", pe_val[1]);
  printf("%-30s%15" PRIu64 "\n", "Branches", pe_val[2]);
  printf("%-30s%15" PRIu64 "\n", "Branch-misses", pe_val[3]);
  printf("%-30s%15.2f%%%s\n", "Branch-miss rate", pe_val[3] * 100.0/pe_val[2], " of all branches");
}

void print_cache(uint64_t *pe_val) {
  printf("%-30s%15" PRIu64 "\n","Cache references:", pe_val[0]);
  printf("%-30s%15" PRIu64 "\n","Cache misses:", pe_val[1]);
  printf("%-30s%15.2f%%%s\n", "Cache misses rate:", pe_val[1]*100.0/pe_val[0], " of all cache refs");
  printf("%-30s%15" PRIu64 "\n", "L1 caches:", pe_val[2]);
  printf("%-30s%15" PRIu64 "\n", "L1 cache misses", pe_val[3]);
  printf("%-30s%15.2f%%%s\n", "L1 cache misse rate:", pe_val[3]*100.0/pe_val[2], " of all L1 reads");
}

void print_tlb(uint64_t *pe_val) {
  printf("%-30s%15" PRIu64 "\n","Cache references:", pe_val[0]);
  printf("%-30s%15" PRIu64 "\n","Cache misses:", pe_val[1]);
  printf("%-30s%15.2f%%%s\n", "Cache misses rate:", pe_val[1]*100.0/pe_val[0], " of all cache refs");
  printf("%-30s%15" PRIu64 "\n", "TLB cache read: ", pe_val[2]);
  printf("%-30s%15" PRIu64 "\n", "TLB cache read misses", pe_val[3]);
  printf("%-30s%15.2f%%%s\n", "TLB cache read miss rate", pe_val[3]*100.0/pe_val[2], " of all TLB reads");
}

void print_results(uint64_t *pe_val, COUNTERS mode) {
  switch(mode) {
    case COUNTERS::INSTR:  
      print_instruction(pe_val); 
      break;
    case COUNTERS::BRANCH: 
      print_branch(pe_val); 
      break;
    case COUNTERS::CACHE:  
      print_cache(pe_val); 
      break;
    case COUNTERS::TLB:    
      print_tlb(pe_val); 
      break;
    case COUNTERS::NONE:
      break;
    default:
      invalid_profile_mode();
  }
}

COUNTERS configure_counters(perf_event_attr* pe) {
  COUNTERS mode = COUNTERS::NONE;
  
  if (getenv("PROFILE_INSTRUCTIONS") &&
      !strcmp(getenv("PROFILE_INSTRUCTIONS"),"1")) {

    if (mode != COUNTERS::NONE) {
      invalid_profile_mode();
    }

    mode = COUNTERS::INSTR;
    // Configure the group of PMUs to count
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    configure_event(&pe[2], PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND);
    configure_event(&pe[3], PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
  }

  if (getenv("PROFILE_BRANCHES") &&
     !strcmp(getenv("PROFILE_BRANCHES"),"1")) {

    if (mode != COUNTERS::NONE) {
      invalid_profile_mode();
    }

    mode = COUNTERS::BRANCH;
    // Configure the group of PMUs to count
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_BPU);
    configure_event(&pe[2], PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    configure_event(&pe[3], PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
  }

  if (getenv("PROFILE_L1_CACHES") &&
     !strcmp(getenv("PROFILE_L1_CACHES"),"1")) {

    if (mode != COUNTERS::NONE) {
      invalid_profile_mode();
    }

    mode = COUNTERS::CACHE;
    // Configure the group of PMUs to count
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    configure_event(&pe[2], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_L1D) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    configure_event(&pe[3], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_L1D) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
  }

  if (getenv("PROFILE_TLB") &&
     !strcmp(getenv("PROFILE_TLB"),"1")) {

    if (mode != COUNTERS::NONE) {
      invalid_profile_mode();
    }

    mode = COUNTERS::TLB;
    // Configure the group of PMUs to count
    configure_event(&pe[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    configure_event(&pe[1], PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    configure_event(&pe[2], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_DTLB) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    configure_event(&pe[3], PERF_TYPE_HW_CACHE, (PERF_COUNT_HW_CACHE_DTLB) |
                                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
  }

  return mode;
}

