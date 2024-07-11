// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops

    float normalize_time = 0;            // in nanoseconds
    unsigned normalize_count = 0;        // # of normalizations
    float pq_preprocess_time = 0;        // in nanoseconds
    unsigned pq_preprocess_count = 0;    // # of pq_preprocess
    float medoid_selection_time = 0;     // in nano seconds
    unsigned medoid_selection_count = 0; // # of medoid_selection
    float search_time = 0;               // in nanoseconds
    unsigned search_count = 0;           // # of searches
    float beam_search_time = 0;          // in nanoseconds
    unsigned beam_search_count = 0;      // # of beam_search
    float frontier_load_time = 0;        // in nanoseconds
    unsigned frontier_load_count = 0;    // # of frontier_load
    float cache_search_time = 0;         // in nanoseconds
    unsigned cache_search_count = 0;     // # of cache_search
    float frontier_search_time = 0;      // in nanoseconds
    unsigned frontier_search_count = 0;  // # of frontier_search
    float data_process_time = 0;         // in nanoseconds
    unsigned data_process_count = 0;     // # of data_process
};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
