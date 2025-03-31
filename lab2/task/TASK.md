# Lab2

##Task 1

Replace reduce_gpu_one_block function in ReduceTest::RunHIP method with your implementation
Reduction function on cpu can be done by [std::reduce](https://en.cppreference.com/w/cpp/algorithm/reduce) 
and [std::acctunuate](https://en.cppreference.com/w/cpp/algorithm/accumulate)
In std::accumulate execution order is fixed

##Task 2

Replace function exclusive_scan in exclusive_gpu method with your implementation
More about  scan can be found [here](https://gfxcourses.stanford.edu/cs149/fall24/lecture/dataparallel/slide_12) 
Scan function on cpu can be done by [std::exclusive_scan](https://en.cppreference.com/w/cpp/algorithm/exclusive_scan) 
Scan function on gpu can be done by [thrust::exclusive_scan](https://nvidia.github.io/cccl/thrust/api/function_group__prefixsums_1ga8dbe92b545e14800f567c69624238d17.html)


##Task 3

Replace function thrust::exclusive_scan_by_key in segmented_scan_gpu method with your implementation
More about  segmented scan can be found [here](thrust::exclusive_scan_by_key)
Scan function on gpu can be done by [thrust::exclusive_scan_by_key](https://nvidia.github.io/cccl/thrust/api/function_group__segmentedprefixsums_1ga0b299a0668efddd3c581d6753a29e98f.html)


##Task 4

Replace function thrust::merge_by_key in group_by_gpu method with your implementation
More about scatter/gather can be found [here](https://gfxcourses.stanford.edu/cs149/fall24/lecture/dataparallel/slide_34)
Group by function on gpu can be done by [thrust::merge_by_key](https://nvidia.github.io/cccl/thrust/api/function_group__merging_1ga2937abe3e9b8bdd1873f3d323bcb1cd0.html)

##Task 5

Replace function thrust::sort_by_key in sort_by_key_gpu method with your implementation
More about sort can be found [here](https://gfxcourses.stanford.edu/cs149/fall24/lecture/dataparallel/slide_37)
Sort function on cpu can be done by [std::sort](https://en.cppreference.com/w/cpp/algorithm/sort)
Sort function on gpu can be done by [thrust::sort_by_key](https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1ga667333ee2e067bb7da3fb1b8ab6d348c.html)

##Task 6

Replace function histogram_gpu in HistogramTest::RunHIP method with your implementation
More about histogram can be found [here](https://gfxcourses.stanford.edu/cs149/fall24/lecture/dataparallel/slide_47)

