#ifndef ORB_LITE_LOOP_CLOSING_HPP
#define ORB_LITE_LOOP_CLOSING_HPP

#include "Map.hpp"
#include "Atlas.hpp"
#include "BoW.hpp"
#include "KeyFrameDatabase.hpp"

namespace orb_lite {

class LoopClosing {
public:
    LoopClosing(Atlas* pAtlas, KeyFrameDatabase* pDB, Vocabulary* pVoc);

    // Returns true if a loop is detected
    bool detectLoop(KeyFrame* pCurrentKF, KeyFrame*& pLoopKF);

    // Corrects the loop (Pose Graph Optimization)
    void correctLoop(KeyFrame* pCurrentKF, KeyFrame* pLoopKF);

private:
    Atlas* mpAtlas;
    KeyFrameDatabase* mpDB;
    Vocabulary* mpVoc;
};

} // namespace orb_lite

#endif
