#include "Vision.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include "Map.hpp"
#include "BoW.hpp"

namespace orb_lite {

static int bit_pattern_31[1024];
static bool pattern_initialized = false;

void initPattern() {
    if (pattern_initialized) return;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(-15, 15);
    for (int i = 0; i < 1024; i++) {
        bit_pattern_31[i] = dis(gen);
    }
    pattern_initialized = true;
}

void detectFAST(const Image& img, std::vector<KeyPoint>& kps, int threshold, bool nonmax) {
    kps.clear();
    for (int y = 3; y < img.rows - 3; y++) {
        for (int x = 3; x < img.cols - 3; x++) {
            unsigned char p = img.data[y * img.cols + x];
            int count = 0;
            int circle_offsets[16][2] = {{0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},{0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}};
            
            bool brighter = true, darker = true;
            for(int i=0; i<16; i++) {
                unsigned char val = img.data[(y + circle_offsets[i][1]) * img.cols + (x + circle_offsets[i][0])];
                if (val <= p + threshold) brighter = false;
                if (val >= p - threshold) darker = false;
                if (!brighter && !darker) break;
            }
            if (brighter || darker) {
                kps.push_back({(float)x, (float)y, 0, 0});
            }
        }
    }
}

void computeORB(const Image& img, std::vector<KeyPoint>& kps, std::vector<Descriptor>& descriptors) {
    initPattern();
    descriptors.clear();
    for (const auto& kp : kps) {
        Descriptor d = {0};
        int x = (int)kp.x, y = (int)kp.y;
        if (x < 16 || x >= img.cols - 16 || y < 16 || y >= img.rows - 16) {
            descriptors.push_back(d);
            continue;
        }
        for (int i = 0; i < 256; i++) {
            int x1 = x + bit_pattern_31[i*4];
            int y1 = y + bit_pattern_31[i*4+1];
            int x2 = x + bit_pattern_31[i*4+2];
            int y2 = y + bit_pattern_31[i*4+3];
            if (img.data[y1 * img.cols + x1] < img.data[y2 * img.cols + x2]) {
                d.val[i / 32] |= (1 << (i % 32));
            }
        }
        descriptors.push_back(d);
    }
}

void matchDescriptors(const std::vector<Descriptor>& d1, const std::vector<Descriptor>& d2, 
                      std::vector<std::pair<int, int>>& matches) {
    matches.clear();
    for (size_t i = 0; i < d1.size(); i++) {
        int best_dist = 60;
        int best_idx = -1;
        for (size_t j = 0; j < d2.size(); j++) {
            int dist = 0;
            for (int k = 0; k < 8; k++) {
                dist += __builtin_popcount(d1[i].val[k] ^ d2[j].val[k]);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = (int)j;
            }
        }
        if (best_idx >= 0) matches.push_back({(int)i, best_idx});
    }
}

void calculateHammingDistance(const Descriptor& d1, const Descriptor& d2, int& distance) {
    distance = 0;
    for (int i = 0; i < 8; i++) {
        distance += __builtin_popcount(d1.val[i] ^ d2.val[i]);
    }
}

int SearchByBoW(KeyFrame* pKF, const std::vector<Descriptor>& descriptors, const FeatureVector& fv, std::vector<std::pair<int, int>>& matches) {
    matches.clear();
    const FeatureVector& kfFeatVec = pKF->featVec;
    
    int nmatches = 0;
    auto kfIt = kfFeatVec.begin();
    auto fIt = fv.begin();
    
    while(kfIt != kfFeatVec.end() && fIt != fv.end()) {
        if(kfIt->first == fIt->first) {
            const auto& kfIndices = kfIt->second;
            const auto& fIndices = fIt->second;
            
            for(unsigned int iKF : kfIndices) {
                int mpId = pKF->mapPointIds[iKF];
                if(mpId < 0) continue;
                
                int bestDist = 70;
                int bestIdxF = -1;
                
                for(unsigned int iF : fIndices) {
                    int dist;
                    calculateHammingDistance(descriptors[iF], pKF->descriptors[iKF], dist);
                    if(dist < bestDist) {
                        bestDist = dist;
                        bestIdxF = iF;
                    }
                }
                
                if(bestIdxF >= 0) {
                    matches.push_back({bestIdxF, mpId});
                    nmatches++;
                }
            }
            kfIt++;
            fIt++;
        } else if(kfIt->first < fIt->first) {
            kfIt++;
        } else {
            fIt++;
        }
    }
    return nmatches;
}

// DLT Triangulation (Standard in ORB-SLAM3)
Vec3 triangulate(const Vec3 &p1_unprot, const Vec3 &p2_unprot, const Mat4x4 &T1w, const Mat4x4 &T2w) {
    Mat4x4 Tw1, Tw2;
    invert4x4(T1w, Tw1);
    invert4x4(T2w, Tw2);
    Vec3 c1 = {Tw1.m[3], Tw1.m[7], Tw1.m[11]};
    Vec3 c2 = {Tw2.m[3], Tw2.m[7], Tw2.m[11]};
    Vec3 v1 = {
        Tw1.m[0]*p1_unprot.x + Tw1.m[1]*p1_unprot.y + Tw1.m[2]*p1_unprot.z,
        Tw1.m[4]*p1_unprot.x + Tw1.m[5]*p1_unprot.y + Tw1.m[6]*p1_unprot.z,
        Tw1.m[8]*p1_unprot.x + Tw1.m[9]*p1_unprot.y + Tw1.m[10]*p1_unprot.z
    };
    Vec3 v2 = {
        Tw2.m[0]*p2_unprot.x + Tw2.m[1]*p2_unprot.y + Tw2.m[2]*p2_unprot.z,
        Tw2.m[4]*p2_unprot.x + Tw2.m[5]*p2_unprot.y + Tw2.m[6]*p2_unprot.z,
        Tw2.m[8]*p2_unprot.x + Tw2.m[9]*p2_unprot.y + Tw2.m[10]*p2_unprot.z
    };
    
    Vec3 w0 = {c1.x - c2.x, c1.y - c2.y, c1.z - c2.z};
    double a = dot(v1, v1), b = dot(v1, v2), c = dot(v2, v2), d = dot(v1, w0), e = dot(v2, w0);
    double denom = a * c - b * b;
    if (std::abs(denom) < 1e-12) return {0, 0, 0};
    double t1 = (b * e - c * d) / denom;
    double t2 = (a * e - b * d) / denom;
    if (t1 <= 0 || t2 <= 0) return {0,0,0};
    return {(c1.x + t1*v1.x + c2.x + t2*v2.x)*0.5, (c1.y + t1*v1.y + c2.y + t2*v2.y)*0.5, (c1.z + t1*v1.z + c2.z + t2*v2.z)*0.5};
}


void matchDescriptorsGrid(const std::vector<KeyPoint>& kps1, const std::vector<Descriptor>& desc1, 
                           const std::vector<KeyPoint>& kps2, const std::vector<Descriptor>& desc2,
                           std::vector<std::pair<int, int>>& matches, float radius) {
    matches.clear();
    if (kps1.empty() || kps2.empty()) return;

    static int grid[48][64][15]; 
    static int grid_count[48][64];
    static std::vector<std::pair<int, int>> used_cells;
    
    // Clear only used cells from previous call
    for (auto& cell : used_cells) grid_count[cell.first][cell.second] = 0;
    used_cells.clear();

    for (int i = 0; i < (int)kps1.size(); i++) {
        int gx = (int)(kps1[i].x / 10.0);
        int gy = (int)(kps1[i].y / 10.0);
        if (gx >= 0 && gx < 64 && gy >= 0 && gy < 48) {
            if (grid_count[gy][gx] == 0) used_cells.push_back({gy, gx});
            if (grid_count[gy][gx] < 15) {
                grid[gy][gx][grid_count[gy][gx]++] = i;
            }
        }
    }

    int grid_radius = (int)(radius / 10.0) + 1;
    for (int i = 0; i < (int)kps2.size(); i++) {
        int gx_c = (int)(kps2[i].x / 10.0);
        int gy_c = (int)(kps2[i].y / 10.0);
        int best_idx = -1;
        int best_dist = 65;

        for (int gy = gy_c - grid_radius; gy <= gy_c + grid_radius; gy++) {
            if (gy < 0 || gy >= 48) continue;
            for (int gx = gx_c - grid_radius; gx <= gx_c + grid_radius; gx++) {
                if (gx < 0 || gx >= 64) continue;
                for (int j = 0; j < grid_count[gy][gx]; j++) {
                    int idx1 = grid[gy][gx][j];
                    int d;
                    calculateHammingDistance(desc1[idx1], desc2[i], d);
                    if (d < best_dist) {
                        best_dist = d;
                        best_idx = idx1;
                    }
                }
            }
        }
        if (best_idx != -1 && best_dist < 45) {
            matches.push_back({best_idx, i});
        }
    }
}

} // namespace orb_lite
