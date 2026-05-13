#ifndef ORB_LITE_IMU_HPP
#define ORB_LITE_IMU_HPP

#include "Math.hpp"
#include <vector>

namespace orb_lite {

struct ImuSample {
    Vec3 acc;
    Vec3 gyro;
    double timestamp;
};

struct Preintegrated {
    double dt = 0;
    Vec3 dP = {0,0,0};
    Vec3 dV = {0,0,0};
    Quaternion dR = {1,0,0,0};

    // Jacobians wrt ba, bg
    Mat3x3 dP_dba = {0,0,0, 0,0,0, 0,0,0};
    Mat3x3 dP_dbg = {0,0,0, 0,0,0, 0,0,0};
    Mat3x3 dV_dba = {0,0,0, 0,0,0, 0,0,0};
    Mat3x3 dV_dbg = {0,0,0, 0,0,0, 0,0,0};
    Mat3x3 dR_dbg = {0,0,0, 0,0,0, 0,0,0};

    void update(const Vec3& acc, const Vec3& gyro, double dt_step, const Vec3& ba, const Vec3& bg);
};

struct NavState {
    Vec3 p;
    Vec3 v;
    Quaternion q;
    Vec3 ba;
    Vec3 bg;
};

// --- Gravity Alignment ---
Quaternion alignGravity(const std::vector<ImuSample>& samples);

} // namespace orb_lite

#endif
