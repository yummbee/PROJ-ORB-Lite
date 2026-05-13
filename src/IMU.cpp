#include "IMU.hpp"
#include <cmath>

namespace orb_lite {

static const double eps = 1e-4;

void Preintegrated::update(const Vec3& acc, const Vec3& gyro, double dt_step, const Vec3& ba, const Vec3& bg) {
    Vec3 a = {acc.x - ba.x, acc.y - ba.y, acc.z - ba.z};
    Vec3 w = {gyro.x - bg.x, gyro.y - bg.y, gyro.z - bg.z};

    Mat3x3 R = qToMat33(dR);
    
    dP.x += dV.x * dt_step + 0.5 * (R.m[0] * a.x + R.m[1] * a.y + R.m[2] * a.z) * dt_step * dt_step;
    dP.y += dV.y * dt_step + 0.5 * (R.m[3] * a.x + R.m[4] * a.y + R.m[5] * a.z) * dt_step * dt_step;
    dP.z += dV.z * dt_step + 0.5 * (R.m[6] * a.x + R.m[7] * a.y + R.m[8] * a.z) * dt_step * dt_step;

    dV.x += (R.m[0] * a.x + R.m[1] * a.y + R.m[2] * a.z) * dt_step;
    dV.y += (R.m[3] * a.x + R.m[4] * a.y + R.m[5] * a.z) * dt_step;
    dV.z += (R.m[6] * a.x + R.m[7] * a.y + R.m[8] * a.z) * dt_step;

    // 3. Update Rotation
    double ang_x = w.x * dt_step;
    double ang_y = w.y * dt_step;
    double ang_z = w.z * dt_step;
    double d2 = ang_x*ang_x + ang_y*ang_y + ang_z*ang_z;
    double d = std::sqrt(d2);

    Mat3x3 deltaR;
    Mat3x3 W = skew({(float)ang_x, (float)ang_y, (float)ang_z});

    if (d < eps) {
        deltaR = mat33Add(Mat3x3::identity(), W);
    } else {
        double s = std::sin(d);
        double c = std::cos(d);
        deltaR = mat33Add(Mat3x3::identity(), mat33Add(mat33Scale(W, s / d), mat33Scale(mat33Mul(W, W), (1.0 - c) / d2)));
    }

    dR = qNormalize(qMul(dR, mat33ToQuat(deltaR)));
    dt += dt_step;
}

Quaternion alignGravity(const std::vector<ImuSample>& samples) {
    if (samples.empty()) return {1, 0, 0, 0};
    Vec3 mean_acc = {0,0,0};
    for (const auto& s : samples) {
        mean_acc.x += s.acc.x;
        mean_acc.y += s.acc.y;
        mean_acc.z += s.acc.z;
    }
    mean_acc.x /= samples.size();
    mean_acc.y /= samples.size();
    mean_acc.z /= samples.size();

    // Standard gravity vector in world frame (Z up)
    Vec3 g_w = {0, 0, 1.0};
    // Gravity vector in IMU frame
    Vec3 g_i = mean_acc;
    double mag = std::sqrt(g_i.x*g_i.x + g_i.y*g_i.y + g_i.z*g_i.z);
    if (mag < 1e-6) return {1,0,0,0};
    g_i.x /= mag; g_i.y /= mag; g_i.z /= mag;

    // Axis of rotation: v = g_i x g_w
    Vec3 v = { g_i.y * g_w.z - g_i.z * g_w.y,
               g_i.z * g_w.x - g_i.x * g_w.z,
               g_i.x * g_w.y - g_i.y * g_w.x };
    double s = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    double c = g_i.x*g_w.x + g_i.y*g_w.y + g_i.z*g_w.z;

    // Use axis-angle to create rotation from IMU to World
    if (s < 1e-6) {
        if (c > 0) return {1,0,0,0};
        else return {0,1,0,0}; // 180 deg
    }
    
    double angle = std::acos(c);
    v.x /= s; v.y /= s; v.z /= s;
    return axisAngleToQuat(v, angle);
}

} // namespace orb_lite
