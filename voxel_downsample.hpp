#ifndef VOXEL_DOWNSAMPLE_HPP
#define VOXEL_DOWNSAMPLE_HPP

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

struct PointBin {
    uint64_t voxel_key;
    size_t original_index;
};

inline uint64_t encode_voxel(
    double x, double y, double z, 
    double min_x, double min_y, double min_z, 
    double voxel_size
) {
    int64_t ix = static_cast<int64_t>(std::floor((x - min_x) / voxel_size));
    int64_t iy = static_cast<int64_t>(std::floor((y - min_y) / voxel_size));
    int64_t iz = static_cast<int64_t>(std::floor((z - min_z) / voxel_size));
    
    ix += 1048576;
    iy += 1048576;
    iz += 1048576;
    
    return (static_cast<uint64_t>(ix & 0x1FFFFF) << 42) |
           (static_cast<uint64_t>(iy & 0x1FFFFF) << 21) |
           (static_cast<uint64_t>(iz & 0x1FFFFF));
}

inline void cpp_voxel_downsample_tracked(
    const double* points, 
    size_t num_points, 
    double voxel_size,
    std::vector<double>& out_points,
    std::vector<int64_t>& out_indices
) {
    if (num_points == 0 || voxel_size <= 0.0) return;

    double min_x = points[0], max_x = points[0];
    double min_y = points[1], max_y = points[1];
    double min_z = points[2], max_z = points[2];
    
    for (size_t i = 1; i < num_points; ++i) {
        double x = points[i * 3];
        double y = points[i * 3 + 1];
        double z = points[i * 3 + 2];
        
        if (x < min_x) min_x = x; else if (x > max_x) max_x = x;
        if (y < min_y) min_y = y; else if (y > max_y) max_y = y;
        if (z < min_z) min_z = z; else if (z > max_z) max_z = z;
    }

    double span_x = (max_x - min_x) / voxel_size;
    double span_y = (max_y - min_y) / voxel_size;
    double span_z = (max_z - min_z) / voxel_size;
    
    if (span_x >= 1048575.0 || span_y >= 1048575.0 || span_z >= 1048575.0) {
        throw std::runtime_error(
            "Point cloud bounding box dimensions (" + 
            std::to_string(span_x) + "x, " + std::to_string(span_y) + "y, " + std::to_string(span_z) + "z grid units) "
            "exceed the max 21-bit limit (1,048,575 units) for the current voxel size: " + std::to_string(voxel_size)
        );
    }

    std::vector<PointBin> bins(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        bins[i].voxel_key = encode_voxel(points[i * 3], points[i * 3 + 1], points[i * 3 + 2], min_x, min_y, min_z, voxel_size);
        bins[i].original_index = i;
    }

    std::sort(bins.begin(), bins.end(), [](const PointBin& a, const PointBin& b) {
        return a.voxel_key < b.voxel_key;
    });

    size_t start = 0;
    while (start < num_points) {
        size_t end = start;
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

        while (end < num_points && bins[end].voxel_key == bins[start].voxel_key) {
            size_t idx = bins[end].original_index;
            sum_x += points[idx * 3];
            sum_y += points[idx * 3 + 1];
            sum_z += points[idx * 3 + 2];
            end++;
        }

        size_t count = end - start;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;
        double avg_z = sum_z / count;

        double min_dist_sq = std::numeric_limits<double>::max();
        int64_t chosen_idx = -1;

        for (size_t i = start; i < end; ++i) {
            size_t idx = bins[i].original_index;
            double dx = points[idx * 3] - avg_x;
            double dy = points[idx * 3 + 1] - avg_y;
            double dz = points[idx * 3 + 2] - avg_z;
            double dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                chosen_idx = static_cast<int64_t>(idx);
            }
        }

        out_points.push_back(points[chosen_idx * 3]);
        out_points.push_back(points[chosen_idx * 3 + 1]);
        out_points.push_back(points[chosen_idx * 3 + 2]);
        out_indices.push_back(chosen_idx);

        start = end;
    }
}

#endif // VOXEL_DOWNSAMPLE_HPP
