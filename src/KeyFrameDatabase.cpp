#include "KeyFrameDatabase.hpp"
#include <algorithm>
#include <set>
#include <iostream>

namespace orb_lite {

KeyFrameDatabase::KeyFrameDatabase(const Vocabulary& voc) : mpVoc(&voc) {
    mvInvertedFile.resize(voc.size());
}

void KeyFrameDatabase::add(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lock(mMutex);
    // For every word detected in this KeyFrame...
    for(auto it = pKF->bowVec.begin(); it != pKF->bowVec.end(); it++) {
        WordId wid = it->first;
        // Add this KeyFrame to the list of frames that contain this word
        mvInvertedFile[wid].push_back(pKF);
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

#include <unordered_map>

std::vector<KeyFrame*> KeyFrameDatabase::detectRelocalizationCandidates(const BowVector& bow) {
    std::vector<KeyFrame*> candidates;
    
    // SAFETY CHECK: If motion blur caused 0 FAST corners, don't waste CPU.
    if (bow.empty()) return candidates; 

    // USE AN UNORDERED MAP: Uses the absolute memory pointer as the unique key. 
    // This prevents Map 1 KF #0 and Map 2 KF #0 from colliding!
    std::unordered_map<KeyFrame*, int> kfShareCount;
    int maxCommonWords = 0;

    {
        std::unique_lock<std::mutex> lock(mMutex);
        
        // CATCH THE UNSIZED DB TRAP
        if (mvInvertedFile.empty()) {
            std::cout << "CRITICAL: mvInvertedFile is empty! Did you call resize(mpVoc->size()) in the constructor?" << std::endl;
            return candidates;
        }

        for(auto it = bow.begin(); it != bow.end(); it++) {
            WordId wid = it->first;
            if(wid >= mvInvertedFile.size()) continue;
            
            for(KeyFrame* pKF : mvInvertedFile[wid]) {
                kfShareCount[pKF]++;
                if (kfShareCount[pKF] > maxCommonWords) {
                    maxCommonWords = kfShareCount[pKF];
                }
            }
        }
    }
    
    if(kfShareCount.empty()) return candidates;

    // Filter by max shared words
    int minCommonWords = (int)(maxCommonWords * 0.8);
    
    std::vector<std::pair<double, KeyFrame*>> scoreKFs;
    for(const auto& pair : kfShareCount) {
        KeyFrame* pKF = pair.first;
        int sharedWords = pair.second;
        
        if(sharedWords >= minCommonWords) {
            double s = mpVoc->score(bow, pKF->bowVec);
            // Ignore random background noise matches
            if (s > 0.015) { 
                scoreKFs.push_back({s, pKF});
            }
        }
    }
    
    // Sort descending by score
    std::sort(scoreKFs.rbegin(), scoreKFs.rend());
    
    for(auto& p : scoreKFs) {
        candidates.push_back(p.second);
        if(candidates.size() >= 10) break;
    }
    
    return candidates;
}

std::vector<KeyFrame*> KeyFrameDatabase::detectLoopCandidates(KeyFrame* pKF, double minScore) {
    std::unordered_map<KeyFrame*, int> kfShareCount;
    std::set<KeyFrame*> candidateSet;
    std::vector<KeyFrame*> candidates;
    
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for(auto it = pKF->bowVec.begin(); it != pKF->bowVec.end(); it++) {
            WordId wid = it->first;
            if(wid >= mvInvertedFile.size()) continue;
            for(KeyFrame* pCandidate : mvInvertedFile[wid]) {
                // Ensure unique tracking via memory address, not potentially colliding IDs
                if(pCandidate->pMap == pKF->pMap && std::abs(pCandidate->id - pKF->id) < 10) continue;
                
                kfShareCount[pCandidate]++;
                candidateSet.insert(pCandidate);
            }
        }
    }
    
    if(candidateSet.empty()) {
        return candidates;
    }

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
