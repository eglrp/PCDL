#include<iostream>
#include<vector>

void findNearestPoints(
    float* points,
    unsigned int point_num,
    float* dists,
    unsigned int* idxs
);

int main()
{

    float pts[]={
    1.f,1.f,1.f,
    2.f,1.f,1.f,
    0.f,1.f,1.f
    };
    std::vector<float> points(pts,pts+9);
    for(int i=0;i<points.size();i++)
    {
        std::cout<<points[i]<<" "<< &points[i] <<std::endl;

    }
    std::vector<float> dists(3);
    std::vector<unsigned int> idxs(3);
    findNearestPoints(points.data(),3,dists.data(),idxs.data());

    for(unsigned int i=0;i<3;i++)
    {
        std::cout<<i<<"th point is nearest to "<<idxs[i]<<"th point with dist "<<dists[i]<< std::endl;
    }
    return 0;
}