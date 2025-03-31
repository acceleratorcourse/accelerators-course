/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_CONFIG_H_IN
#define GUARD_CONFIG_H_IN

#define MIOPEN_BACKEND_OPENCL 0
#define MIOPEN_BACKEND_HIP 1
#define MIOPEN_MODE_NOGPU 0
#define MIOPEN_USE_ROCBLAS 1
#define MIOPEN_USE_HIPBLASLT 1
#define MIOPEN_USE_ROCTRACER 1
#define MIOPEN_BUILD_DEV 0
#define MIOPEN_GPU_SYNC 0

#define MIOPEN_ENABLE_SQLITE 1
#define MIOPEN_ENABLE_SQLITE_KERN_CACHE 1
#define MIOPEN_DEBUG_FIND_DB_CACHING 1
#define MIOPEN_USE_COMGR 1
#define MIOPEN_USE_HIPRTC 1
#define MIOPEN_USE_HIP_KERNELS 1
#define MIOPEN_DISABLE_USERDB 0
#define MIOPEN_EMBED_DB 0
#define BUILD_SHARED_LIBS 1
#define MIOPEN_DISABLE_SYSDB 0
#define MIOPEN_LOG_FUNC_TIME_ENABLE 0
#define MIOPEN_ENABLE_SQLITE_BACKOFF 1
#define MIOPEN_USE_MLIR 1
#define MIOPEN_USE_COMPOSABLEKERNEL 1
#define MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK 1
#define MIOPEN_ENABLE_AI_KERNEL_TUNING 1
#define MIOPEN_HIP_COMPILER_HAS_OPTION_OFFLOAD_UNIFORM_BLOCK 1
#define MIOPEN_WORKAROUND_USE_BOOST_FILESYSTEM 0

// "_PACKAGE_" to avoid name contentions: the macros like
// HIP_VERSION_MAJOR are defined in hip_version.h.
// clang-format off
#define HIP_PACKAGE_VERSION_MAJOR 6
#define HIP_PACKAGE_VERSION_MINOR 1
#define HIP_PACKAGE_VERSION_PATCH 40091
// clang-format on

// clang-format off
#define MIOPEN_AMD_COMGR_VERSION_MAJOR 2
#define MIOPEN_AMD_COMGR_VERSION_MINOR 7
#define MIOPEN_AMD_COMGR_VERSION_PATCH 0
// clang-format on

// Truncation rounding or (default) rounding to nearest even (RNE) is enabled.
// This switch controls two related but different aspects of MIOpen behavior:
// 1.  How host code performs conversions of float to bfloat16, important only
//     for testing.
// 2.  How BF16 kernels (which are kind of mixed-precision now and expected to
//     remain in the future)  perform final conversion (and rounding) of FP32
//     to BF16 results. This affects the main functionality of the library.
#define MIOPEN_USE_RNE_BFLOAT16 1
#define MIOPEN_FP8_IEEE_EXPONENT_BIAS 0
#define MIOPEN_FP8_CLIPPING 1

// clang-format off
#define MIOPEN_DEFAULT_FIND_MODE DynamicHybrid

// clang-format on
#define MIOPEN_OFFLINE_COMPILER_PATHS_V2 0
#if MIOPEN_OFFLINE_COMPILER_PATHS_V2

// APIs to get MIOPEN Compiler Paths generated using ROCM Base install Path
// This will read the ROCM Base Install Path if it is already generated
// or (if required), will generate the MIOPEN Compiler Flag Path Values
// and return the value.
const char* getAMDGCNAssemblerPath();
const char* getOpenCLCompilerPath();
const char* getHIPCompilerPath();
const char* getOffloadBundlerBinPath();

#define MIOPEN_AMDGCN_ASSEMBLER getAMDGCNAssemblerPath()
#define HIP_OC_COMPILER getOpenCLCompilerPath()
#define MIOPEN_HIP_COMPILER getHIPCompilerPath()
#define MIOPEN_OFFLOADBUNDLER_BIN getOffloadBundlerBinPath()
#else
#define MIOPEN_AMDGCN_ASSEMBLER "/opt/rocm/llvm/bin/clang"
#define HIP_OC_COMPILER "/opt/rocm/bin/amdclang"
#define MIOPEN_HIP_COMPILER "/opt/rocm/llvm/bin/clang++"
#define MIOPEN_OFFLOADBUNDLER_BIN "/opt/rocm/llvm/bin/clang-offload-bundler"
#endif // MIOPEN_OFFLINE_COMPILER_PATHS_V2

#define MIOPEN_CACHE_DIR "~/.cache/miopen/"

#define MIOPEN_USE_SQLITE_PERFDB 0

#define MIOPEN_USE_GEMM (MIOPEN_USE_ROCBLAS || MIOPEN_USE_HIPBLASLT)

// Usage of "defined" operator within macro expansion is undefined behavior,
// so "defined(NDEBUG)" cannot be used there... unlike the following macro:
#ifdef NDEBUG
#define MIOPEN_NDEBUG 1
#else
#define MIOPEN_NDEBUG 0
#endif

// Installable builds are those which aren't intended for debugging.
// We damp down some diagnostic messages (Error -> Warning) etc.
#define MIOPEN_INSTALLABLE (MIOPEN_NDEBUG && !MIOPEN_BUILD_DEV)

#define MIOPEN_ALLOC_BUFFERS 0

#ifndef HIP_PACKAGE_VERSION_MAJOR
#define HIP_PACKAGE_VERSION_MAJOR 0
#endif
#ifndef HIP_PACKAGE_VERSION_MINOR
#define HIP_PACKAGE_VERSION_MINOR 0
#endif
#ifndef HIP_PACKAGE_VERSION_PATCH
#define HIP_PACKAGE_VERSION_PATCH 0
#endif
// 3 decimal digits for major and minor, 6 digits for patch number.
// Max number is 999,999,999999 == 0xE8,D4A5,0FFF that fits into 64-bit math.
#if HIP_PACKAGE_VERSION_MAJOR > 999 || HIP_PACKAGE_VERSION_MAJOR > 999 || \
    HIP_PACKAGE_VERSION_PATCH > 999999
#error "Too big HIP version number(s)"
#endif
#define HIP_PACKAGE_VERSION_FLAT                                                   \
    ((HIP_PACKAGE_VERSION_MAJOR * 1000ULL + HIP_PACKAGE_VERSION_MINOR) * 1000000 + \
     HIP_PACKAGE_VERSION_PATCH)

#if MIOPEN_USE_ROCBLAS
// clang-format off
#define MIOPEN_ROCBLAS_VERSION_MAJOR 4
#define MIOPEN_ROCBLAS_VERSION_MINOR 1
#define MIOPEN_ROCBLAS_VERSION_PATCH 0
// clang-format on
#ifndef MIOPEN_ROCBLAS_VERSION_MAJOR
#define MIOPEN_ROCBLAS_VERSION_MAJOR 0
#endif
#ifndef MIOPEN_ROCBLAS_VERSION_MINOR
#define MIOPEN_ROCBLAS_VERSION_MINOR 0
#endif
#ifndef MIOPEN_ROCBLAS_VERSION_PATCH
#define MIOPEN_ROCBLAS_VERSION_PATCH 0
#endif
// 3 decimal digits for each number; max fits into 32 bits.
#if MIOPEN_ROCBLAS_VERSION_MAJOR > 999 || MIOPEN_ROCBLAS_VERSION_MAJOR > 999 || \
    MIOPEN_ROCBLAS_VERSION_PATCH > 999
#error "Too big ROCBLAS version number(s)"
#endif
#define MIOPEN_ROCBLAS_VERSION_FLAT                                                \
    ((MIOPEN_ROCBLAS_VERSION_MAJOR * 1000 + MIOPEN_ROCBLAS_VERSION_MINOR) * 1000 + \
     MIOPEN_ROCBLAS_VERSION_PATCH)
#endif // MIOPEN_USE_ROCBLAS

#if MIOPEN_USE_HIPBLASLT
// clang-format off
#define MIOPEN_HIPBLASLT_VERSION_MAJOR 0
#define MIOPEN_HIPBLASLT_VERSION_MINOR 7
#define MIOPEN_HIPBLASLT_VERSION_PATCH 0
// clang-format on
#ifndef MIOPEN_HIPBLASLT_VERSION_MAJOR
#define MIOPEN_HIPBLASLT_VERSION_MAJOR 0
#endif
#ifndef MIOPEN_HIPBLASLT_VERSION_MINOR
#define MIOPEN_HIPBLASLT_VERSION_MINOR 0
#endif
#ifndef MIOPEN_HIPBLASLT_VERSION_PATCH
#define MIOPEN_HIPBLASLT_VERSION_PATCH 0
#endif
// 3 decimal digits for each number; max fits into 32 bits.
#if MIOPEN_HIPBLASLT_VERSION_MAJOR > 999 || MIOPEN_HIPBLASLT_VERSION_MAJOR > 999 || \
    MIOPEN_HIPBLASLT_VERSION_PATCH > 999
#error "Too big HIPBLASLT version number(s)"
#endif
#define MIOPEN_HIPBLASLT_VERSION_FLAT                                                  \
    ((MIOPEN_HIPBLASLT_VERSION_MAJOR * 1000 + MIOPEN_HIPBLASLT_VERSION_MINOR) * 1000 + \
     MIOPEN_HIPBLASLT_VERSION_PATCH)
#endif // MIOPEN_USE_HIPBLASLT

/// WORKAROUND_BOOST_ISSUE_392
/// Workaround for https://github.com/boostorg/config/issues/392#issuecomment-1109889533
/// See also https://github.com/ROCm/MIOpen/pull/1490#issuecomment-1109928102,
/// https://github.com/ROCm/MIOpen/pull/1543
/// TODO: Remove the W/A as soon we switch to the properly fixed boost.
#if MIOPEN_BACKEND_HIP
#include <hip/hip_version.h>
#endif

#define MIOPEN_GOLDEN_DB_VERSION 20

#endif // GUARD_CONFIG_H_IN
