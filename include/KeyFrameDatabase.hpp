#ifndef ORB_LITE_KEYFRAME_DATABASE_HPP
#define ORB_LITE_KEYFRAME_DATABASE_HPP

#include "Map.hpp"
#include "BoW.hpp"
#include <vector>
#include <list>
#include <mutex>

namespace orb_lite {

class KeyFrameDatabase {
public:
    KeyFrameDatabase(const Vocabulary& voc);

    void add(KeyFrame* pKF);
    void erase(KeyFrame* pKF);
    void clear();

    std::vector<KeyFrame*> detectLoopCandidates(KeyFrame* pKF, double minScore);
    std::vector<KeyFrame*> detectRelocalizationCandidates(const BowVector& bow);

protected:
    const Vocabulary* mpVoc;
    std::vector<std::list<KeyFrame*>> mvInvertedFile;
    std::mutex mMutex;
};

} // namespace orb_lite

#endif
