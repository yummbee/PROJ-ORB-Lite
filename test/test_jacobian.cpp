#include "Optimizer.hpp"
#include <iostream>
#include <vector>

namespace orb_lite {

void testOptimizerJacobian() {
    Camera cam = {500, 500, 320, 240};
    NavState state;
    state.p = {0, 0, 0};
    state.q = {1, 0, 0, 0};

    MapPoint mp;
    mp.pos = {0.5, 0.5, 5.0}; // Point in front of camera
    
    // Project true position
    Vec3 Pc = mp.pos;
    Vec2 uv = {cam.fx * Pc.x / Pc.z + cam.cx, cam.fy * Pc.y / Pc.z + cam.cy};
    
    std::cout << "--- Optimizer Jacobian Numerical Test ---" << std::endl;
    
    double delta_val = 1e-6;
    for (int i = 0; i < 6; i++) {
        double d[6] = {0};
        d[i] = delta_val;
        
        // Perturb state
        NavState s_plus = state;
        Mat3x3 R_wc = qToMat33(state.q);
        s_plus.p.x += (R_wc.m[0]*d[0] + R_wc.m[1]*d[1] + R_wc.m[2]*d[2]);
        s_plus.p.y += (R_wc.m[3]*d[0] + R_wc.m[4]*d[1] + R_wc.m[5]*d[2]);
        s_plus.p.z += (R_wc.m[6]*d[0] + R_wc.m[7]*d[1] + R_wc.m[8]*d[2]);
        Quaternion dq = {1.0, d[3]*0.5, d[4]*0.5, d[5]*0.5};
        s_plus.q = qNormalize(qMul(state.q, dq));
        
        // Project perturbed
        Mat3x3 R_cw_plus = mat33Transpose(qToMat33(s_plus.q));
        Vec3 t_cw_plus = {
            -(R_cw_plus.m[0]*s_plus.p.x + R_cw_plus.m[1]*s_plus.p.y + R_cw_plus.m[2]*s_plus.p.z),
            -(R_cw_plus.m[3]*s_plus.p.x + R_cw_plus.m[4]*s_plus.p.y + R_cw_plus.m[5]*s_plus.p.z),
            -(R_cw_plus.m[6]*s_plus.p.x + R_cw_plus.m[7]*s_plus.p.y + R_cw_plus.m[8]*s_plus.p.z)
        };
        Vec3 Pc_plus = {
            R_cw_plus.m[0]*mp.pos.x + R_cw_plus.m[1]*mp.pos.y + R_cw_plus.m[2]*mp.pos.z + t_cw_plus.x,
            R_cw_plus.m[3]*mp.pos.x + R_cw_plus.m[4]*mp.pos.y + R_cw_plus.m[5]*mp.pos.z + t_cw_plus.y,
            R_cw_plus.m[6]*mp.pos.x + R_cw_plus.m[7]*mp.pos.y + R_cw_plus.m[8]*mp.pos.z + t_cw_plus.z
        };
        Vec2 uv_plus = {cam.fx * Pc_plus.x / Pc_plus.z + cam.cx, cam.fy * Pc_plus.y / Pc_plus.z + cam.cy};
        
        double J_num_x = (uv_plus.x - uv.x) / delta_val;
        double J_num_y = (uv_plus.y - uv.y) / delta_val;
        
        // Analytic (using standard Right-Increment on T_wc)
        // J = d_proj/d_Pc * [-I | [Pc]x]
        double invz = 1.0 / Pc.z;
        double invz2 = invz * invz;
        double Jp[2][3] = {
            {cam.fx * invz, 0, -cam.fx * Pc.x * invz2},
            {0, cam.fy * invz, -cam.fy * Pc.y * invz2}
        };
        
        double J_full[2][6] = {0};
        // Translation part: -Jp
        J_full[0][0] = -Jp[0][0]; J_full[0][1] = -Jp[0][1]; J_full[0][2] = -Jp[0][2];
        J_full[1][0] = -Jp[1][0]; J_full[1][1] = -Jp[1][1]; J_full[1][2] = -Jp[1][2];
        // Rotation part: Jp * [Pc]x
        // [Pc]x = [ 0 -z  y ]
        //         [ z  0 -x ]
        //         [-y  x  0 ]
        J_full[0][3] = Jp[0][1]*Pc.z - Jp[0][2]*Pc.y;
        J_full[0][4] = -Jp[0][0]*Pc.z + Jp[0][2]*Pc.x;
        J_full[0][5] = Jp[0][0]*Pc.y - Jp[0][1]*Pc.x;
        
        J_full[1][3] = Jp[1][1]*Pc.z - Jp[1][2]*Pc.y;
        J_full[1][4] = -Jp[1][0]*Pc.z + Jp[1][2]*Pc.x;
        J_full[1][5] = Jp[1][0]*Pc.y - Jp[1][1]*Pc.x;
        
        std::cout << "Axis " << i << ": Num=(" << J_num_x << ", " << J_num_y 
                  << ") Ana=(" << J_full[0][i] << ", " << J_full[1][i] << ")" << std::endl;
    }
}

} // namespace orb_lite

int main() {
    orb_lite::testOptimizerJacobian();
    return 0;
}
