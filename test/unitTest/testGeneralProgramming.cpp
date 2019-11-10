#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>
int main(int argc, char **argv)
{
    std::vector<float> a;
    for (int i = 0; i < 16; ++i)
    {
        a.push_back(i);
    }

    const int row = 4, cols = 4;
    Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> b(a.data(), row, cols);
    std::cout << b << std::endl;
}