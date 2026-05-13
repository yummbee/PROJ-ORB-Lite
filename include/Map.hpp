#ifndef ORB_LITE_MAP_HPP
#define ORB_LITE_MAP_HPP

#include "Math.hpp"
#include "Vision.hpp"
#include "BoW.hpp"
#include <vector>
#include <deque>

namespace orb_lite {

struct MapPoint {
    Vec3 pos;
    Descriptor descriptor;
    int observedCount = 0;
    int refKfId = -1; // Reference KeyFrame for relative position
    bool isBad = false;
    class Map* pMap = nullptr;
};

struct KeyFrame {
    int id;
    Mat4x4 pose; // World to Camera
    std::vector<KeyPoint> kps;
    std::vector<Descriptor> descriptors;
    std::vector<int> mapPointIds; // -1 if no map point
    
    BowVector bowVec;
    FeatureVector featVec;
    
    int parentId = -1;
    std::vector<int> neighbors; // IDs of connected KeyFrames
    class Map* pMap = nullptr;
};

class Map {
public:
    std::deque<MapPoint> points;
    std::deque<KeyFrame> keyframes;

    int addPoint(const Vec3& pos, const Descriptor& desc, int refKfId = -1);
    int addKeyframe(KeyFrame& kf);

    void save(const char* filename) const;
    void load(const char* filename);
    void saveCSV(const char* filename) const;
    
    // Map Maintenance
    void cleanMap();
    void mergePoints(int id1, int id2);
};

} // namespace orb_lite

#endif
