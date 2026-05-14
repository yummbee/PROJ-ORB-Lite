#ifndef ORB_LITE_OPTIMIZER_HPP
#define ORB_LITE_OPTIMIZER_HPP

#include "Map.hpp"
#include "Tracking.hpp"
#include <vector>
#include <unordered_set>

namespace orb_lite {

struct OptimizationState {
    Mat4x4 pose;
    Vec3 p;
    Vec3 v;
    Vec3 ba;
    Vec3 bg;
};

class Optimizer {
public:
    static void bundleAdjustment(Map& map, const FeatureExtractor& extractor, int numIterations = 10);
    static void localBundleAdjustment(Map& map, const FeatureExtractor& extractor, int currentKfId, int windowSize = 5);
    
    // Pose-only optimization (similar to TrackLocalMap in ORB-SLAM3)
static int poseOptimization(std::vector<std::pair<int, int>>& matches, 
                                  const std::vector<KeyPoint>& kps,
                                  const Camera& cam,
                                  NavState& state,
                                  Map& map,
                                  const FeatureExtractor& extractor);

    // Global Pose Graph Optimization for Loop Closing
    static void optimizeEssentialGraph(Map& map, KeyFrame* pCurrentKF, KeyFrame* pLoopKF, const Mat4x4& Tloop);

    // Similarity transformation (Horn's Method) for Map Merging
    static Sim3 computeSim3(const std::vector<Vec3>& pts1, const std::vector<Vec3>& pts2);

private:
    static void optimizeSingleMapPoint(MapPoint& mp, Map& map, const Camera& cam, int mp_idx, const FeatureExtractor& extractor, const std::unordered_set<int>& fixedKfs);
};

// --- Helper for small linear systems ---
template<int N>
bool solveLinear(const double* H, const double* B, double* delta);

} // namespace orb_lite

#endif
