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

void applyLowPassFilter(const Image& src, Image& dst) {
    // Skip 1-pixel border to avoid bounds checks
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += src.data[(y + dy) * src.cols + (x + dx)];
                }
            }
            dst.data[y * dst.cols + x] = (uint8_t)(sum / 9);
        }
    }
}

bool isFASTCorner(const Image& img, int x, int y, int threshold) {
    unsigned char p = img.data[y * img.cols + x];
    int circle_offsets[16][2] = {{0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},{0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}};
    
    bool brighter = true, darker = true;
    for(int i=0; i<16; i++) {
        unsigned char val = img.data[(y + circle_offsets[i][1]) * img.cols + (x + circle_offsets[i][0])];
        if (val <= p + threshold) brighter = false;
        if (val >= p - threshold) darker = false;
        if (!brighter && !darker) break;
    }
    return brighter || darker;
}

int calculateCornerScore(const Image& img, int cx, int cy) {
    // Bounds check to prevent segfaults on the edge of the image
    if (cx < 3 || cy < 3 || cx >= img.cols - 3 || cy >= img.rows - 3) return 0;
    
    int score = 0;
    int center_val = img.data[cy * img.cols + cx];
    
    // Sum the contrast of the 5x5 patch surrounding the corner
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            if (dx == 0 && dy == 0) continue;
            int val = img.data[(cy + dy) * img.cols + (cx + dx)];
            score += std::abs(val - center_val);
        }
    }
    return score;
}

void applyGridNMS(const Image& img, const std::vector<KeyPoint>& raw_kps, std::vector<KeyPoint>& filtered_kps, int cell_size) {
    // 1. Calculate how many grid cells cover our image
    int grid_cols = (img.cols / cell_size) + 1;
    int grid_rows = (img.rows / cell_size) + 1;

    // 2. Create the Grid (Tracking the champion of each cell)
    std::vector<std::vector<KeyPoint>> grid_best_kp(grid_rows, std::vector<KeyPoint>(grid_cols));
    std::vector<std::vector<int>> grid_best_score(grid_rows, std::vector<int>(grid_cols, -1));

    // 3. The Battle Royale
    for (const auto& kp : raw_kps) {
        int gx = kp.x / cell_size;
        int gy = kp.y / cell_size;

        if (gx < 0 || gx >= grid_cols || gy < 0 || gy >= grid_rows) continue;

        // Score the raw point
        int score = calculateCornerScore(img, kp.x, kp.y);

        // If it's the strongest corner seen in this specific cell so far, it takes the throne
        if (score > grid_best_score[gy][gx]) {
            grid_best_score[gy][gx] = score;
            grid_best_kp[gy][gx] = kp;
        }
    }

    // 4. Collect the Champions
    filtered_kps.clear();
    for (int gy = 0; gy < grid_rows; gy++) {
        for (int gx = 0; gx < grid_cols; gx++) {
            // Check the score threshold to ensure the cell wasn't just filled with flat noise
            // 150 is a strong baseline. Raise it to 200 if you still get shimmer.
            if (grid_best_score[gy][gx] > 150) { 
                filtered_kps.push_back(grid_best_kp[gy][gx]);
            }
        }
    }
}

void detectFAST(const Image& img, std::vector<KeyPoint>& kps, int threshold, bool nonmax) {
    kps.clear();
    for (int y = 16; y < img.rows - 16; y++) {
        for (int x = 16; x < img.cols - 16; x++) {
            if (isFASTCorner(img, x, y, threshold)) {
                kps.push_back({(float)x, (float)y, 0, 0, 0});
            }
        }
    }
}

void extractFeaturesGrid(const Image& img, std::vector<KeyPoint>& kps, int threshold) {
    kps.clear();
    const int CELL_SIZE = 30;
    int grid_cols = img.cols / CELL_SIZE;
    int grid_rows = img.rows / CELL_SIZE;
    
    for(int gy = 0; gy < grid_rows; gy++) {
        for(int gx = 0; gx < grid_cols; gx++) {
            int cell_start_x = gx * CELL_SIZE;
            int cell_start_y = gy * CELL_SIZE;
            
            int best_score = -1;
            KeyPoint best_kp;
            bool found_corner_in_cell = false;

            for(int y = std::max(16, cell_start_y); y < std::min(img.rows - 16, cell_start_y + CELL_SIZE); y++) {
                for(int x = std::max(16, cell_start_x); x < std::min(img.cols - 16, cell_start_x + CELL_SIZE); x++) {
                    if(isFASTCorner(img, x, y, threshold)) {
                        int score = calculateCornerScore(img, x, y);
                        if(score > best_score) {
                            best_score = score;
                            best_kp.x = (float)x;
                            best_kp.y = (float)y;
                            best_kp.response = (float)score;
                            found_corner_in_cell = true;
                        }
                    }
                }
            }
            if(found_corner_in_cell && best_score > 150) { 
                kps.push_back(best_kp);
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

void resizeBilinear(const Image& src, Image& dst) {
    float x_ratio = ((float)(src.cols - 1)) / dst.cols;
    float y_ratio = ((float)(src.rows - 1)) / dst.rows;

    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            // Calculate exact sub-pixel source coordinates
            float x = j * x_ratio;
            float y = i * y_ratio;
            
            // Get the integer coordinates of the 4 surrounding pixels
            int x_floor = (int)x;
            int y_floor = (int)y;
            int x_ceil = std::min(x_floor + 1, src.cols - 1);
            int y_ceil = std::min(y_floor + 1, src.rows - 1);

            // Calculate the fractional distances
            float x_diff = x - x_floor;
            float y_diff = y - y_floor;

            // Retrieve the 4 surrounding pixel values using the Image::at() method
            uint8_t a = src.at(y_floor, x_floor);
            uint8_t b = src.at(y_floor, x_ceil);
            uint8_t c = src.at(y_ceil, x_floor);
            uint8_t d = src.at(y_ceil, x_ceil);

            // Perform Bilinear Interpolation
            float top = a * (1.0f - x_diff) + b * x_diff;
            float bottom = c * (1.0f - x_diff) + d * x_diff;
            dst.at(i, j) = (uint8_t)(top * (1.0f - y_diff) + bottom * y_diff);
        }
    }
}

FeatureExtractor::FeatureExtractor() {
    mvScaleFactor.resize(nLevels);
    mvInvScaleFactor.resize(nLevels);
    mvLevelSigma2.resize(nLevels);
    mvInvLevelSigma2.resize(nLevels);

    mvScaleFactor[0] = 1.0f;
    mvInvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    mvInvLevelSigma2[0] = 1.0f;

    for (int i = 1; i < nLevels; i++) {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }
}

void FeatureExtractor::computePyramid(const Image& base_img, std::vector<Image>& imagePyramid) {
    imagePyramid.resize(nLevels);
    mvImageMemory.resize(nLevels);
    
    imagePyramid[0] = base_img;
    // Level 0 doesn't need to own memory if base_img is managed elsewhere

    for (int level = 1; level < nLevels; level++) {
        float scale = mvInvScaleFactor[level];
        int scaled_w = (int)std::round(base_img.cols * scale);
        int scaled_h = (int)std::round(base_img.rows * scale);
        
        // Allocate memory for this level
        mvImageMemory[level].resize(scaled_w * scaled_h);
        
        // Set up the Image wrapper
        imagePyramid[level].data = mvImageMemory[level].data();
        imagePyramid[level].cols = scaled_w;
        imagePyramid[level].rows = scaled_h;
        imagePyramid[level].stride = scaled_w;

        resizeBilinear(imagePyramid[level - 1], imagePyramid[level]);
    }
}

void FeatureExtractor::extract(const Image& base_img, std::vector<KeyPoint>& all_kps, std::vector<Descriptor>& all_descs) {
    all_kps.clear();
    all_descs.clear();

    // 0. Pre-filter Level 0 to kill ISO noise
    mvBlurredImage.resize(base_img.cols * base_img.rows);
    Image blurred_img = base_img;
    blurred_img.data = mvBlurredImage.data();
    applyLowPassFilter(base_img, blurred_img);

    std::vector<Image> imagePyramid;
    computePyramid(blurred_img, imagePyramid);

    for (int level = 0; level < nLevels; level++) {
        std::vector<KeyPoint> level_kps;
        std::vector<Descriptor> level_descs;
        
        // 1. Run raw FAST extraction
        std::vector<KeyPoint> raw_kps;
        detectFAST(imagePyramid[level], raw_kps, 20, true);

        // 2. Cull the herd (Spatial Grid NMS)
        applyGridNMS(imagePyramid[level], raw_kps, level_kps, 30);

        // 3. Compute descriptors on the survivors
        computeORB(imagePyramid[level], level_kps, level_descs);

        // 3. Project coordinates back to Level 0
        for (size_t i = 0; i < level_kps.size(); i++) {
            KeyPoint kp = level_kps[i];
            
            // Multiply by the scale factor to return to Level 0 coordinates
            kp.x = kp.x * mvScaleFactor[level];
            kp.y = kp.y * mvScaleFactor[level];
            kp.octave = level;

            all_kps.push_back(kp);
            all_descs.push_back(level_descs[i]);
        }
    }
}

} // namespace orb_lite
