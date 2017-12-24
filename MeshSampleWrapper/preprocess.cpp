//
// Created by pal on 17-12-13.
//
#include "preprocess.h"
#include "mesh_sample.h"

#include <cmath>
#include <random>

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


void rotatePointCloud(std::vector<double>& point_cloud)
{
    double phi=uniform_rand()*2*M_PI;
    double cos_val=cos(phi);
    double sin_val=sin(phi);
    for(size_t i=0;i<point_cloud.size()/3;i++)
    {
        double x0=point_cloud[i*3];
        double y0=point_cloud[i*3+1];
        double x=x0*cos_val-y0*sin_val;
        double y=x0*sin_val+y0*cos_val;
        point_cloud[i*3]=x;
        point_cloud[i*3+1]=y;
    }
}


void addNoise(std::vector<double>& point_cloud,double noise_level)
{
    std::random_device rd;
    std::normal_distribution<double> dis(0.0,noise_level);
    std::default_random_engine gen(rd());
    for(auto it=point_cloud.begin();it!=point_cloud.end();it++)
    {
        (*it)+=dis(gen);
    }
}

void normalize(std::vector<double>& point_cloud)
{
    double min_x=point_cloud[0],max_x=point_cloud[0],
            min_y=point_cloud[1],max_y=point_cloud[1],
            min_z=point_cloud[2],max_z=point_cloud[2];
    for(size_t i=1;i<point_cloud.size()/3;i++)
    {
        double x0=point_cloud[i*3];
        double y0=point_cloud[i*3+1];
        double z0=point_cloud[i*3+2];
        min_x=std::min(x0,min_x);
        min_y=std::min(y0,min_y);
        min_z=std::min(z0,min_z);
        max_x=std::max(x0,max_x);
        max_y=std::max(y0,max_y);
        max_z=std::max(z0,max_z);
    }

    double center_x=(min_x+max_x)/2.0;
    double center_y=(min_y+max_y)/2.0;
    double center_z=(min_z+max_z)/2.0;
    double max_dist=0;
    for(size_t i=0;i<point_cloud.size()/3;i++)
    {
        point_cloud[i*3]-=center_x;
        point_cloud[i*3+1]-=center_y;
        point_cloud[i*3+2]-=center_z;
        max_dist=std::max(max_dist,point_cloud[i*3]*point_cloud[i*3]+
                                   point_cloud[i*3+1]*point_cloud[i*3+1]+
                                   point_cloud[i*3+2]*point_cloud[i*3+2]);
    }
    max_dist=sqrt(max_dist);
    for(auto it=point_cloud.begin();it!=point_cloud.end();it++)
    {
        (*it)/=max_dist;
    }
}


void computeDists(std::vector<double>& point_cloud,std::vector<double>& dists)
{
    size_t pt_num=point_cloud.size()/3;
    dists.resize(pt_num*pt_num);
    for(size_t i=0;i<pt_num;i++)
    {
        dists[i*pt_num+i]=0;
        double x0=point_cloud[i*3];
        double y0=point_cloud[i*3+1];
        double z0=point_cloud[i*3+2];
        for(size_t j=i+1;j<pt_num;j++)
        {
            double x1=point_cloud[j*3];
            double y1=point_cloud[j*3+1];
            double z1=point_cloud[j*3+2];

            double dist=std::sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));

            dists[i*pt_num+j]=dist;
            dists[j*pt_num+i]=dist;
        }
    }
}