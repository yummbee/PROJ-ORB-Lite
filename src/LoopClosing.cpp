#include "LoopClosing.hpp"
#include "Vision.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <map>

namespace orb_lite {

LoopClosing::LoopClosing(Atlas* pAtlas, KeyFrameDatabase* pDB, Vocabulary* pVoc)
    : mpAtlas(pAtlas), mpDB(pDB), mpVoc(pVoc) {}

bool LoopClosing::detectLoop(KeyFrame* pCurrentKF, KeyFrame*& pLoopKF) {
    if(!mpVoc || !mpDB) return false;

    // Detect loop candidates from all maps
    auto candidates = mpDB->detectLoopCandidates(pCurrentKF, 0.01);
    
    for(KeyFrame* pKF : candidates) {
        // Skip neighbors in the same map
        if (pKF->pMap == pCurrentKF->pMap) {
            bool isNeighbor = false;
            for(int nid : pCurrentKF->neighbors) {
                if(nid == pKF->id) { isNeighbor = true; break; }
            }
            if(isNeighbor) continue;
        }

        std::vector<std::pair<int, int>> matches;
        SearchByBoW(pKF, pCurrentKF->descriptors, pCurrentKF->featVec, matches);
        
        if (matches.size() >= 20) {
            NavState state;
            state.p = {pKF->pose.m[3], pKF->pose.m[7], pKF->pose.m[11]};
            state.q = mat33ToQuat(mat33Transpose(mat33ExtractRotation(pKF->pose)));
            
            Camera cam = {700.0, 700.0, 320.0, 240.0};
            Optimizer::poseOptimization(matches, pCurrentKF->kps, cam, state, *pKF->pMap);
            
            if (matches.size() >= 15) {
                std::cout << "Loop Detection: CONFIRMED loop with KeyFrame " << pKF->id << " in map " << pKF->pMap << std::endl;
                pLoopKF = pKF;
                return true;
            }
        }
    }
    
    return false;
}

void LoopClosing::correctLoop(KeyFrame* pCurrentKF, KeyFrame* pLoopKF) {
    std::cout << "Loop Closing: Correcting KF " << pCurrentKF->id << " using KF " << pLoopKF->id << std::endl;
    
    // Compute relative correction T_loop_curr
    // We want P_loop = T_loop_curr * P_curr
    // Since we optimized 'state' in detectLoop to match pLoopKF, we can use that.
    // However, for simplicity, we compute it from the confirmed poses.
    
    Mat4x4 T_curr_world; invert4x4(pCurrentKF->pose, T_curr_world);
    Mat4x4 T_loop_curr = mat44Mul(pLoopKF->pose, T_curr_world);

    if (pCurrentKF->pMap != pLoopKF->pMap) {
        std::cout << "Loop Closing: Cross-map loop detected. Merging maps..." << std::endl;
        mpAtlas->mergeMaps(pLoopKF->pMap, pCurrentKF->pMap, T_loop_curr);
    } else {
        std::cout << "Loop Closing: Intra-map loop detected. Running Essential Graph PGO..." << std::endl;
        Optimizer::optimizeEssentialGraph(*pCurrentKF->pMap, pCurrentKF, pLoopKF, T_loop_curr);
    }
}

} // namespace orb_lite
