//
// Created by pal on 17-12-12.
//

#ifndef MESHSAMPLE_BATCH_FILE_IO_H
#define MESHSAMPLE_BATCH_FILE_IO_H

#include <string>
#include <vector>

void readBatchFileHead(
        const std::string& filename,
        unsigned long& model_num
);

void readBatchFileData(
        const std::string& file_name,
        unsigned int index,
        std::vector<double>& vertexes,
        std::vector<unsigned int>& faces,
        std::vector<double>& areas,
        double& total_area,
        unsigned int& category
);

#endif //MESHSAMPLE_BATCH_FILE_IO_H
