#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

// Layout is fixed - NCHW
//
template <typename T>
struct TensorDescriptor
{
    TensorDescriptor(){};

    TensorDescriptor(const std::vector<int>& lens_in) : lens(lens_in)
    {
        strides.resize(lens.size(), 0);
        strides.back() = 1;
        std::partial_sum(
            lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<int>());
    }

    const std::vector<int>& GetLengths() const { return lens; };

    size_t GetElementSize() const
    {
        return std::accumulate(lens.begin(), lens.end(), (int)1, std::multiplies<int>());
    }

    template <class... Ts>
    std::size_t GetIndex(Ts... is) const
    {
        return this->GetIndex({static_cast<int>(is)...});
    }

    std::size_t GetIndex(std::initializer_list<int> l) const
    {
        return std::inner_product(
            l.begin() + 1, l.end(), strides.begin(), static_cast<std::size_t>(*(l.begin())));
    };

    //    std::size_t GetIndex(std::initializer_list<int> l) const {
    //        return std::inner_product(
    //          l.begin(), l.end(), strides.begin(), 0);
    //    };

    std::string ToString() const
    {
        std::string result;
        if(this->lens.empty())
            return result;
        for(auto i : this->lens)
        {
            result += std::to_string(i) + ", ";
        }
        return result.substr(0, result.length() - 2);
    }

    friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
    {
        LogRange(stream << "{", t.lens, ", ") << "}, ";
        LogRange(stream << "{", t.strides, ", ") << "}, ";

        return stream;
    }

    std::vector<int> lens;
    std::vector<int> strides;
};

template <class Range>
std::ostream& LogRange(std::ostream& os, Range&& r, std::string delim)
{
    bool first = true;
    for(auto&& x : r)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << x;
    }
    return os;
}

#endif // TENSOR_HPP
