#ifndef ORB_LITE_VISION_HPP
#define ORB_LITE_VISION_HPP

#include "Math.hpp"
#include <vector>
#include <cstdint>

namespace orb_lite {

struct KeyPoint {
    float x, y;
    float angle;
    float response;
    int octave;
};

struct Image {
    uint8_t* data;
    int cols;
    int rows;
    int stride;

    uint8_t& at(int y, int x) { return data[y * stride + x]; }
    const uint8_t& at(int y, int x) const { return data[y * stride + x]; }
};

struct Descriptor {
    uint32_t val[8]; // 256 bits
};

// --- FAST-9 Detector ---
void detectFAST(const Image& img, std::vector<KeyPoint>& kps, int threshold, bool nms);

// --- Orientation ---
float computeOrientation(const Image& img, float x, float y);

// --- Rotated BRIEF ---
void computeORB(const Image& img, std::vector<KeyPoint>& kps, std::vector<Descriptor>& descriptors);

// --- Matching ---
void matchDescriptors(const std::vector<Descriptor>& desc1, const std::vector<Descriptor>& desc2, std::vector<std::pair<int, int>>& matches);
void matchDescriptorsGrid(const std::vector<KeyPoint>& kps1, const std::vector<Descriptor>& desc1, 
                           const std::vector<KeyPoint>& kps2, const std::vector<Descriptor>& desc2,
                           std::vector<std::pair<int, int>>& matches, float radius = 50.0f);
void calculateHammingDistance(const Descriptor& d1, const Descriptor& d2, int& distance);

// --- Reconstruction ---
Vec3 triangulate(const Vec3& p1_unprot, const Vec3& p2_unprot, const Mat4x4& T1w, const Mat4x4& T2w);

// --- BoW-Accelerated Matching ---
struct KeyFrame;
class FeatureVector;
int SearchByBoW(KeyFrame* pKF, const std::vector<Descriptor>& descriptors, const FeatureVector& fv, std::vector<std::pair<int, int>>& matches);

} // namespace orb_lite

#endif
