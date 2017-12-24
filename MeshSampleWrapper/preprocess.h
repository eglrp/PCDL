//
// Created by pal on 17-12-13.
//

#ifndef MESHSAMPLEWRAPPER_PREPROCESS_H
#define MESHSAMPLEWRAPPER_PREPROCESS_H

#endif //MESHSAMPLEWRAPPER_PREPROCESS_H

#include <vector>



void transformToRelativePolarForm(
        const std::vector<double> &pts,
        std::vector<double> &relative_polar_pts
);

void rotatePointCloud(std::vector<double>& point_cloud);

void addNoise(std::vector<double>& point_cloud,double noise_level);

void normalize(std::vector<double>& point_cloud);

void computeDists(std::vector<double>& point_cloud,std::vector<double>& dists);