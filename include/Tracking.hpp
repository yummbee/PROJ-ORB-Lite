#ifndef ORB_LITE_TRACKING_HPP
#define ORB_LITE_TRACKING_HPP

#include "Math.hpp"
#include "Vision.hpp"
#include "Map.hpp"
#include "Atlas.hpp"
#include "BoW.hpp"
#include "KeyFrameDatabase.hpp"
#include "LoopClosing.hpp"
#include "IMU.hpp"
#include <vector>

namespace orb_lite {

enum TrackingState {
    NO_IMAGES_YET,
    NOT_INITIALIZED,
    OK,
    LOST
};

struct Camera {
    double fx, fy, cx, cy;
    
    Vec2 project(const Vec3& p_cam) const {
        return {fx * p_cam.x / p_cam.z + cx, fy * p_cam.y / p_cam.z + cy};
    }
    
    Vec3 unproject(float x, float y) const {
        return {(x - cx) / fx, (y - cy) / fy, 1.0};
    }
};

struct Tracking {
    Camera cam;
    NavState state;
    Atlas* atlas = nullptr;
    Map* map = nullptr;
    Vocabulary* voc = nullptr;
    KeyFrameDatabase* kf_db = nullptr;
    struct LoopClosing* loop_closer = nullptr;
    
    Tracking() {
        state.p = {0,0,0};
        state.v = {0,0,0};
        state.q = {1,0,0,0};
        state.ba = {0,0,0};
        state.bg = {0,0,0};
    }
    
    TrackingState tracking_state = NO_IMAGES_YET;
    bool is_initialized = false;
    bool imu_calibrated = false;
    std::vector<ImuSample> imu_calibration_buffer;
    double first_imu_time = -1.0;
    double last_frame_time = -1.0;
    
    double last_t = -1.0;
    int frame_count = 0;
    int consecutive_lost_frames = 0;

    // For bootstrapping
    std::vector<KeyPoint> last_kps;
    std::vector<KeyPoint> current_kps;
    std::vector<Descriptor> last_descriptors;
    Mat4x4 last_T_world_cam;

    // Current frame BoW
    BowVector current_bow;
    FeatureVector current_feat_vec;

    std::vector<std::pair<int, int>> tracked_map_indices; 
    
    // Feature grid for fast spatial matching
    static const int GRID_COLS = 64;
    static const int GRID_ROWS = 48;
    std::vector<int> feature_grid[48][64];

    bool track(const Image& img, const std::vector<ImuSample>& imu, double timestamp);
    void predictPose(const std::vector<ImuSample>& imu, double timestamp);
    int addPoint(const Vec3& pos, const Descriptor& desc, int refKfId = -1);
    int addKeyframe(KeyFrame& kf);
    void searchMapPoints(const std::vector<Descriptor>& descriptors, const std::vector<KeyPoint>& kps, std::vector<std::pair<int, int>>& matches);
    
    bool relocalize(const std::vector<Descriptor>& descriptors, const std::vector<KeyPoint>& kps);
};

} // namespace orb_lite

#endif
