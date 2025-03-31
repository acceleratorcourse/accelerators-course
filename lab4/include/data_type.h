#include <miopen/miopen.h>
//#include <cudnn_frontend.h>

template <typename T>
struct to_miopen_data_type
{
};

template <>
struct to_miopen_data_type<double>
{
    static miopenDataType_t get() { return miopenDouble; }
};

template <>
struct to_miopen_data_type<float>
{
    static miopenDataType_t get() { return miopenFloat; }
};

template <>
struct to_miopen_data_type<half_float::half>
{
    static miopenDataType_t get() { return miopenHalf; } // we actually didn't calculate 16bit float
};

template <>
struct to_miopen_data_type<int8_t>
{
    static miopenDataType_t get() { return miopenInt8; }
};

template <>
struct to_miopen_data_type<bfloat16>
{
    static miopenDataType_t get() { return miopenBFloat16; }
};

/*
namespace cudnn_frontend {
template <typename t>
struct to_cudnn_data_type
{
};

template <>
struct to_cudnn_data_type<double>
{
    static DataType_t get() { return DataType_t::DOUBLE; }
};

template <>
struct to_cudnn_data_type<float>
{
    static DataType_t get() { return DataType_t::DOUBLE; }
};

template <>
struct to_cudnn_data_type<half_float::half>
{
    static DataType_t get() { return DataType_t::DOUBLE; }
};

template <>
struct to_cudnn_data_type<int8_t>
{
    static DataType_t get() { return DataType_t::INT8; }
};

template <>
struct to_cudnn_data_type<bfloat16>
{
    static DataType_t get() { return DataType_t::BFLOAT16; }
};
}
*/

