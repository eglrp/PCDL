//
// Created by pal on 17-12-12.
//

#ifndef MESHSAMPLE_MESH_SAMPLE_H
#define MESHSAMPLE_MESH_SAMPLE_H

#include <vector>

void sampleMeshPoints(
        const std::vector<double>& vertexes,
        const std::vector<unsigned int>& faces,
        const std::vector<double>& cumul_area,
        double total_area,
        unsigned int sample_num,
        std::vector<double>& point_cloud
);

void sampleMeshPoints(
        const std::vector<double>& vertexes,
        const std::vector<unsigned int>& faces,
        const std::vector<double>& cumul_area,
        double total_area,
        unsigned int sample_num,
        std::vector<double>& point_cloud,
        std::vector<double>& normal_cloud
);

double uniform_rand();

#endif //MESHSAMPLE_MESH_SAMPLE_H
