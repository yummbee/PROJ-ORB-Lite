#include "Optimizer.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace orb_lite;

double dist(Vec3 a, Vec3 b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}

int main() {
    std::cout << "Testing Pose Optimization..." << std::endl;

    Camera cam = {700.0, 700.0, 320.0, 240.0};
    
    // 1. Create a true pose
    NavState true_state;
    true_state.p = {1.0, 0.5, -0.2};
    true_state.q = {0.92388, 0, 0.38268, 0}; // 45 deg around Y
    
    // 2. Create synthetic map points
    Map map;
    std::vector<KeyPoint> kps;
    std::vector<std::pair<int, int>> matches;

    Mat3x3 R_wc = qToMat33(true_state.q);
    Mat3x3 R_cw = mat33Transpose(R_wc);
    Vec3 t_cw = {
        -(R_cw.m[0]*true_state.p.x + R_cw.m[1]*true_state.p.y + R_cw.m[2]*true_state.p.z),
        -(R_cw.m[3]*true_state.p.x + R_cw.m[4]*true_state.p.y + R_cw.m[5]*true_state.p.z),
        -(R_cw.m[6]*true_state.p.x + R_cw.m[7]*true_state.p.y + R_cw.m[8]*true_state.p.z)
    };

    for (int i = 0; i < 100; i++) {
        Vec3 pw = {(double)(rand()%100)/10.0 - 5.0, (double)(rand()%100)/10.0 - 5.0, (double)(rand()%100)/10.0 + 2.0};
        
        Vec3 pc = {
            R_cw.m[0]*pw.x + R_cw.m[1]*pw.y + R_cw.m[2]*pw.z + t_cw.x,
            R_cw.m[3]*pw.x + R_cw.m[4]*pw.y + R_cw.m[5]*pw.z + t_cw.y,
            R_cw.m[6]*pw.x + R_cw.m[7]*pw.y + R_cw.m[8]*pw.z + t_cw.z
        };

        if (pc.z > 0.5) {
            Vec2 uv = cam.project(pc);
            if (uv.x > 10 && uv.x < 630 && uv.y > 10 && uv.y < 470) {
                int mp_idx = map.addPoint(pw, Descriptor());
                int kp_idx = (int)kps.size();
                kps.push_back({(float)uv.x, (float)uv.y});
                matches.push_back({kp_idx, mp_idx});
            }
        }
    }

    std::cout << "Generated " << matches.size() << " synthetic observations." << std::endl;

    // 3. Add noise to initial state
    NavState noisy_state = true_state;
    noisy_state.p.x += 0.2;
    noisy_state.p.y -= 0.1;
    noisy_state.q = qNormalize(qMul(noisy_state.q, {0.999, 0.01, -0.02, 0.01}));

    std::cout << "Initial Position Error: " << dist(noisy_state.p, true_state.p) << "m" << std::endl;

    // 4. Optimize
    Optimizer::poseOptimization(matches, kps, cam, noisy_state, map);

    // 5. Check result
    double final_err = dist(noisy_state.p, true_state.p);
    std::cout << "Final Position Error: " << final_err << "m" << std::endl;

    if (final_err < 1e-2) {
        std::cout << "SUCCESS: Pose recovered!" << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Pose optimization failed." << std::endl;
        return 1;
    }
}
