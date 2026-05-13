#include "KeyFrameDatabase.hpp"
#include <algorithm>
#include <set>

namespace orb_lite {

KeyFrameDatabase::KeyFrameDatabase(const Vocabulary& voc) : mpVoc(&voc) {
    mvInvertedFile.resize(voc.size());
}

void KeyFrameDatabase::add(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lock(mMutex);
    for(auto it = pKF->bowVec.begin(); it != pKF->bowVec.end(); it++) {
        mvInvertedFile[it->first].push_back(pKF);
    }
}

void KeyFrameDatabase::erase(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lock(mMutex);
    for(auto it = pKF->bowVec.begin(); it != pKF->bowVec.end(); it++) {
        mvInvertedFile[it->first].remove(pKF);
    }
}

void KeyFrameDatabase::clear() {
    std::unique_lock<std::mutex> lock(mMutex);
    for(auto& l : mvInvertedFile) l.clear();
}

std::vector<KeyFrame*> KeyFrameDatabase::detectRelocalizationCandidates(const BowVector& bow) {
    std::vector<int> kfShareCount(10000, 0); // Assuming < 10000 KFs for now
    std::set<KeyFrame*> candidateSet;
    std::vector<KeyFrame*> candidates;
    
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for(auto it = bow.begin(); it != bow.end(); it++) {
            WordId wid = it->first;
            if(wid >= mvInvertedFile.size()) continue;
            for(KeyFrame* pKF : mvInvertedFile[wid]) {
                if(pKF->id >= (int)kfShareCount.size()) kfShareCount.resize(pKF->id + 1000, 0);
                kfShareCount[pKF->id]++;
                candidateSet.insert(pKF);
            }
        }
    }
    
    if(candidateSet.empty()) return candidates;

    // Filter by max shared words
    int maxCommonWords = 0;
    for(KeyFrame* pKF : candidateSet) {
        if(kfShareCount[pKF->id] > maxCommonWords) maxCommonWords = kfShareCount[pKF->id];
    }

    int minCommonWords = (int)(maxCommonWords * 0.8);
    
    std::vector<std::pair<double, KeyFrame*>> scoreKFs;
    for(KeyFrame* pKF : candidateSet) {
        if(kfShareCount[pKF->id] > minCommonWords) {
            double s = mpVoc->score(bow, pKF->bowVec);
            scoreKFs.push_back({s, pKF});
        }
    }
    
    std::sort(scoreKFs.rbegin(), scoreKFs.rend());
    
    for(auto& p : scoreKFs) {
        candidates.push_back(p.second);
        if(candidates.size() > 10) break;
    }
    
    return candidates;
}

std::vector<KeyFrame*> KeyFrameDatabase::detectLoopCandidates(KeyFrame* pKF, double minScore) {
    std::vector<int> kfShareCount(10000, 0);
    std::set<KeyFrame*> candidateSet;
    std::vector<KeyFrame*> candidates;
    
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for(auto it = pKF->bowVec.begin(); it != pKF->bowVec.end(); it++) {
            WordId wid = it->first;
            if(wid >= mvInvertedFile.size()) continue;
            for(KeyFrame* pCandidate : mvInvertedFile[wid]) {
                if(std::abs(pCandidate->id - pKF->id) < 20) continue;
                if(pCandidate->id >= (int)kfShareCount.size()) kfShareCount.resize(pCandidate->id + 1000, 0);
                kfShareCount[pCandidate->id]++;
                candidateSet.insert(pCandidate);
            }
        }
    }
    
    if(candidateSet.empty()) return candidates;

    std::vector<std::pair<double, KeyFrame*>> scoreKFs;
    for(KeyFrame* pCandidate : candidateSet) {
        double s = mpVoc->score(pKF->bowVec, pCandidate->bowVec);
        if(s >= minScore) {
            scoreKFs.push_back({s, pCandidate});
        }
    }
    
    std::sort(scoreKFs.rbegin(), scoreKFs.rend());
    for(auto& p : scoreKFs) {
        candidates.push_back(p.second);
        if(candidates.size() > 5) break;
    }
    
    return candidates;
}

} // namespace orb_lite
