#include "Math.hpp"
#include "Vision.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "ORB-Lite initialized." << std::endl;
    
    orb_lite::Mat3x3 m = orb_lite::Mat3x3::identity();
    orb_lite::Mat3x3 inv;
    if (orb_lite::invert3x3(m, inv)) {
        std::cout << "Math Kernel: Matrix inversion works." << std::endl;
    }

    std::vector<uint8_t> data(100 * 100, 128);
    orb_lite::Image img = {data.data(), 100, 100, 100};
    std::vector<orb_lite::KeyPoint> kps;
    orb_lite::detectFAST(img, kps, 20, true);
    
    std::cout << "Front-End: FAST detector initialized. Found " << kps.size() << " corners in uniform image." << std::endl;

    return 0;
}
