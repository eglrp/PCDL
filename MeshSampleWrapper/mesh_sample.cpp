//
// Created by pal on 17-12-12.
//

#include "mesh_sample.h"
#include <random>

using namespace std;

std::random_device rd;
std::default_random_engine gen(rd());
std::uniform_real_distribution<double> dis(0.0,1.0);

double uniform_rand()
{
    return dis(gen);
}

inline void sampleTriangle(
        const std::vector<double>& vertexes,
        unsigned int p1,unsigned int p2,unsigned int p3,
        double& sample_x,double& sample_y,double& sample_z
)
{
    double phi=uniform_rand(),theta=uniform_rand();
    if((1.0-phi)*(1.0-phi)+(1.0-theta)*(1.0-theta)<phi*phi+theta*theta)
    {
        phi=1.0-phi;
        theta=1.0-theta;
    }

    double pt1x=vertexes[p1*3];
    double pt1y=vertexes[p1*3+1];
    double pt1z=vertexes[p1*3+2];
    double pt2x=vertexes[p2*3];
    double pt2y=vertexes[p2*3+1];
    double pt2z=vertexes[p2*3+2];
    double pt3x=vertexes[p3*3];
    double pt3y=vertexes[p3*3+1];
    double pt3z=vertexes[p3*3+2];

    double diffx21=pt2x-pt1x;
    double diffy21=pt2y-pt1y;
    double diffz21=pt2z-pt1z;
    double diffx31=pt3x-pt1x;
    double diffy31=pt3y-pt1y;
    double diffz31=pt3z-pt1z;

    sample_x=phi*diffx21+theta*diffx31+pt1x;
    sample_y=phi*diffy21+theta*diffy31+pt1y;
    sample_z=phi*diffz21+theta*diffz31+pt1z;
}


inline void sampleTriangle(
        const std::vector<double>& vertexes,
        unsigned int p1,unsigned int p2,unsigned int p3,
        double& sample_x,double& sample_y,double& sample_z,
        double& normal_x,double& normal_y,double& normal_z
)
{
    double phi=uniform_rand(),theta=uniform_rand();
    if((1.0-phi)*(1.0-phi)+(1.0-theta)*(1.0-theta)<phi*phi+theta*theta)
    {
        phi=1.0-phi;
        theta=1.0-theta;
    }

    double pt1x=vertexes[p1*3];
    double pt1y=vertexes[p1*3+1];
    double pt1z=vertexes[p1*3+2];
    double pt2x=vertexes[p2*3];
    double pt2y=vertexes[p2*3+1];
    double pt2z=vertexes[p2*3+2];
    double pt3x=vertexes[p3*3];
    double pt3y=vertexes[p3*3+1];
    double pt3z=vertexes[p3*3+2];

    double diffx21=pt2x-pt1x;
    double diffy21=pt2y-pt1y;
    double diffz21=pt2z-pt1z;
    double diffx31=pt3x-pt1x;
    double diffy31=pt3y-pt1y;
    double diffz31=pt3z-pt1z;

    sample_x=phi*diffx21+theta*diffx31+pt1x;
    sample_y=phi*diffy21+theta*diffy31+pt1y;
    sample_z=phi*diffz21+theta*diffz31+pt1z;

    normal_x=diffy21*diffz31-diffz21*diffy31;
    normal_y=diffz21*diffx31-diffx21*diffz31;
    normal_z=diffx21*diffy31-diffy21*diffx31;
    double normal_len=sqrt(normal_x*normal_x+normal_y*normal_y+normal_z*normal_z);

    normal_x/=normal_len;
    normal_y/=normal_len;
    normal_z/=normal_len;
}

void sampleMeshPoints(
        const std::vector<double>& vertexes,
        const std::vector<unsigned int>& faces,
        const std::vector<double>& cumul_area,
        double total_area,
        unsigned int sample_num,
        std::vector<double>& point_cloud,
        std::vector<double>& normal_cloud
)
{
    point_cloud.resize(sample_num*3);
    normal_cloud.resize(sample_num*3);
    for(size_t i=0;i<sample_num;i++)
    {
        double rand_area=uniform_rand()*total_area;
        auto it=std::lower_bound(cumul_area.begin(),cumul_area.end(),rand_area);
        long face_index=it-cumul_area.begin();

        sampleTriangle(vertexes,faces[face_index*3],faces[face_index*3+1],faces[face_index*3+2],
                       point_cloud[i*3],point_cloud[i*3+1],point_cloud[i*3+2],
                       normal_cloud[i*3],normal_cloud[i*3+1],normal_cloud[i*3+2]);
    }
}

void sampleMeshPoints(
        const std::vector<double>& vertexes,
        const std::vector<unsigned int>& faces,
        const std::vector<double>& cumul_area,
        double total_area,
        unsigned int sample_num,
        std::vector<double>& point_cloud
)
{
    point_cloud.resize(sample_num*3);
    for(size_t i=0;i<sample_num;i++)
    {
        double rand_area=uniform_rand()*total_area;
        auto it=std::lower_bound(cumul_area.begin(),cumul_area.end(),rand_area);
        long face_index=it-cumul_area.begin();

        sampleTriangle(vertexes,faces[face_index*3],faces[face_index*3+1],faces[face_index*3+2],
                       point_cloud[i*3],point_cloud[i*3+1],point_cloud[i*3+2]);
    }
}

