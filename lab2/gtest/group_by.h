#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/merge.h>

template<typename KeysIterator, typename ValsIterator, typename OutputKeysIterator, typename OutputValsIterator>
thrust::pair<OutputKeysIterator, OutputValsIterator> group_by_gpu(KeysIterator a_keys_begin,
	                                                       KeysIterator a_keys_end,
                                                               KeysIterator b_keys_begin,
			                                       KeysIterator b_keys_end,
                                                               ValsIterator a_vals_begin,
                                                               ValsIterator b_vals_begin,
                                                               OutputKeysIterator result_keys_begin,
                                                               OutputValsIterator result_vals_begin) {
                                                               
    return thrust::merge_by_key(a_keys_begin, a_keys_end, 
		                 b_keys_begin, b_keys_end, 
				 a_vals_begin, 
				 b_vals_begin, 
				 result_keys_begin, 
				 result_vals_begin);
				 
}

