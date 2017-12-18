//
// Created by pal on 17-12-13.
//
#include "preprocess.h"
#include <cmath>

void transformToRelativePolarForm(
        const std::vector<double> &pts,
        std::vector<double> &relative_polar_pts
)
{
    // centralize
    auto pt_num= static_cast<unsigned int>(pts.size()/3);
    double mx=0.0,my=0.0,mz=0.0;
    for(unsigned int i=0;i<pt_num;i++)
    {
        int index=i*3;
        mx+=pts[index];
        my+=pts[index+1];
        mz+=pts[index+2];
    }
    mx/=pt_num;my/=pt_num;mz/=pt_num;

    //convert to polar coordinate
    std::vector<double> polar_pts(pt_num*3);
    for(unsigned int i=0;i<pt_num;i++)
    {
        int index=i*3;
        double rx=pts[index]-mx;
        double ry=pts[index+1]-my;
        double rz=pts[index+2]-mz;

        polar_pts[index]=atan2(ry,rx);
        polar_pts[index+1]=sqrt(ry*ry+rx*rx);
        polar_pts[index+2]=rz;
    }

    //compute relative relationship
    relative_polar_pts.resize(pt_num*(pt_num-1)*5);

    for(unsigned int i=0;i<pt_num;i++)
    {
        int result_index1=i*(pt_num-1)*5;
        int pt_index1=i*3;
        int k=0;
        for(unsigned int j=0;j<pt_num;j++)
        {
            if(i==j)  continue;
            int result_index2=result_index1+k*5;
            int pt_index2=j*3;
            relative_polar_pts[result_index2]=polar_pts[pt_index1]-polar_pts[pt_index2];   // delta theta

            if(relative_polar_pts[result_index2]>M_PI)
                relative_polar_pts[result_index2]-=M_PI;
            else if(relative_polar_pts[result_index2]<-M_PI)
                relative_polar_pts[result_index2]+=M_PI;

            relative_polar_pts[result_index2+1]=polar_pts[pt_index1+1];                    // r1
            relative_polar_pts[result_index2+2]=polar_pts[pt_index2+1];                    // r2
            relative_polar_pts[result_index2+3]=polar_pts[pt_index1+2];                    // h1
            relative_polar_pts[result_index2+4]=polar_pts[pt_index2+2];                    // h2
            k++;
        }
    }
}
