#ifndef ORB_LITE_ATLAS_HPP
#define ORB_LITE_ATLAS_HPP

#include "Map.hpp"
#include <vector>
#include <mutex>

namespace orb_lite {

class Atlas {
public:
    Atlas();
    ~Atlas();

    void createNewMap();
    Map* getCurrentMap();
    std::vector<Map*> getAllMaps();
    
    void mergeMaps(Map* pTargetMap, Map* pCurrentMap, const Mat4x4& T_target_current);

private:
    std::vector<Map*> m_maps;
    Map* m_currentMap;
    std::mutex m_mutex;
};

} // namespace orb_lite

#endif // ORB_LITE_ATLAS_HPP
