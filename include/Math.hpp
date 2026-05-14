#ifndef ORB_LITE_MATH_HPP
#define ORB_LITE_MATH_HPP

#include <cmath>
#include <array>

namespace orb_lite {

struct Vec2 {
    double x, y;
};

struct Vec4 {
    double x, y, z, w;
    double& operator[](int i) { return (&x)[i]; }
    const double& operator[](int i) const { return (&x)[i]; }
};

struct Vec3 {
    double x, y, z;
    double& operator[](int i) { return (&x)[i]; }
    const double& operator[](int i) const { return (&x)[i]; }
};

struct Mat3x3 {
    double m[9]; // Row-major: 0,1,2, 3,4,5, 6,7,8

    double& at(int r, int c) { return m[r * 3 + c]; }
    const double& at(int r, int c) const { return m[r * 3 + c]; }

    const static Mat3x3 zeros() {
        Mat3x3 m;
        for(int i=0; i<9; i++) m.m[i] = 0;
        return m;
    }

    static Mat3x3 identity() {
        return {1, 0, 0, 0, 1, 0, 0, 0, 1};
    }
};

struct Mat4x4 {
    double m[16]; // Row-major

    double& at(int r, int c) { return m[r * 4 + c]; }
    const double& at(int r, int c) const { return m[r * 4 + c]; }

    static Mat4x4 zeros() {
        Mat4x4 m;
        for(int i=0; i<16; i++) m.m[i] = 0;
        return m;
    }

    static Mat4x4 identity() {
        return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }
};

struct Quaternion {
    double w, x, y, z;
};

struct Sim3 {
    Quaternion q;
    Vec3 t;
    double s;

    // Transform a point: P' = s * R * P + t
    // Note: We need qToMat33 and mat33MulVec3 which are defined later.
    // So we'll define the method body after those functions.
    Vec3 transform(const Vec3& p) const;
};

inline double distance_between(const Vec3& p1, const Vec3& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

inline double calculate_angle_diff(const Quaternion& q1, const Quaternion& q2) {
    double d = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
    if (d < 0) d = -d;
    if (d > 1.0) d = 1.0;
    return 2.0 * std::acos(d);
}

// --- Geometric Primitives ---

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double normSq(const Vec3& v) {
    return dot(v, v);
}

inline double norm(const Vec3& v) {
    return std::sqrt(normSq(v));
}

inline Vec3 normalized(const Vec3& v) {
    double n = norm(v);
    if (n < 1e-12) return {0, 0, 0};
    return {v.x / n, v.y / n, v.z / n};
}

// --- Matrix Inversion (Cramer's Rule) ---

inline double det3x3(double m0, double m1, double m2,
                     double m3, double m4, double m5,
                     double m6, double m7, double m8) {
    return m0 * (m4 * m8 - m5 * m7) - m1 * (m3 * m8 - m5 * m6) + m2 * (m3 * m7 - m4 * m6);
}

inline bool invert3x3(const Mat3x3& in, Mat3x3& out) {
    double d = det3x3(in.m[0], in.m[1], in.m[2],
                      in.m[3], in.m[4], in.m[5],
                      in.m[6], in.m[7], in.m[8]);
    if (std::abs(d) < 1e-15) return false;
    double invDet = 1.0 / d;

    out.m[0] = (in.m[4] * in.m[8] - in.m[5] * in.m[7]) * invDet;
    out.m[1] = (in.m[2] * in.m[7] - in.m[1] * in.m[8]) * invDet;
    out.m[2] = (in.m[1] * in.m[5] - in.m[2] * in.m[4]) * invDet;
    out.m[3] = (in.m[5] * in.m[6] - in.m[3] * in.m[8]) * invDet;
    out.m[4] = (in.m[0] * in.m[8] - in.m[2] * in.m[6]) * invDet;
    out.m[5] = (in.m[2] * in.m[3] - in.m[0] * in.m[5]) * invDet;
    out.m[6] = (in.m[3] * in.m[7] - in.m[4] * in.m[6]) * invDet;
    out.m[7] = (in.m[1] * in.m[6] - in.m[0] * in.m[7]) * invDet;
    out.m[8] = (in.m[0] * in.m[4] - in.m[1] * in.m[3]) * invDet;
    return true;
}

inline Mat4x4 mat44Mul(const Mat4x4& a, const Mat4x4& b) {
    Mat4x4 r;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            r.at(i, j) = a.at(i, 0) * b.at(0, j) + a.at(i, 1) * b.at(1, j) + 
                         a.at(i, 2) * b.at(2, j) + a.at(i, 3) * b.at(3, j);
        }
    }
    return r;
}

inline bool invert4x4(const Mat4x4& in, Mat4x4& out) {
    double m[16];
    for(int i=0; i<16; ++i) m[i] = in.m[i];

    double inv[16];
    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
               m[4]  * m[11] * m[14] + 
               m[8]  * m[6]  * m[15] - 
               m[8]  * m[7]  * m[14] - 
               m[12] * m[6]  * m[11] + 
               m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
                m[4]  * m[10] * m[13] +
                m[8]  * m[5] * m[14] - 
                m[8]  * m[6] * m[13] - 
                m[12] * m[5] * m[10] + 
                m[12] * m[6] * m[9];

    double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::abs(det) < 1e-15) return false;

    inv[1] = -m[1]  * m[10] * m[15] + 
               m[1]  * m[11] * m[14] + 
               m[9]  * m[2] * m[15] - 
               m[9]  * m[3] * m[14] - 
               m[13] * m[2] * m[11] + 
               m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
               m[0]  * m[11] * m[13] + 
               m[8]  * m[1] * m[15] - 
               m[8]  * m[3] * m[13] - 
               m[12] * m[1] * m[11] + 
               m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
                m[0]  * m[10] * m[13] - 
                m[8]  * m[1] * m[14] + 
                m[8]  * m[2] * m[13] + 
                m[12] * m[1] * m[10] - 
                m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
               m[0]  * m[7] * m[14] + 
               m[4]  * m[2] * m[15] - 
               m[4]  * m[3] * m[14] - 
               m[12] * m[2] * m[7] + 
               m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
                m[0]  * m[7] * m[13] - 
                m[4]  * m[1] * m[15] + 
                m[4]  * m[3] * m[13] + 
                m[12] * m[1] * m[7] - 
                m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
                m[0]  * m[6] * m[13] + 
                m[4]  * m[1] * m[14] - 
                m[4]  * m[2] * m[13] - 
                m[12] * m[1] * m[6] + 
                m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
               m[1] * m[7] * m[10] + 
               m[5] * m[2] * m[11] - 
               m[5] * m[3] * m[10] - 
               m[9] * m[2] * m[7] + 
               m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
                m[0] * m[7] * m[9] + 
                m[4] * m[1] * m[11] - 
                m[4] * m[3] * m[9] - 
                m[8] * m[1] * m[7] + 
                m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
                m[0] * m[6] * m[9] - 
                m[4] * m[1] * m[10] + 
                m[4] * m[2] * m[9] + 
                m[8] * m[1] * m[6] - 
                m[8] * m[2] * m[5];

    double detInv = 1.0 / det;
    for (int i = 0; i < 16; i++) out.m[i] = inv[i] * detInv;
    return true;
}

// --- Quaternion Math ---

inline Quaternion qMul(const Quaternion& a, const Quaternion& b) {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

inline Quaternion axisAngleToQuat(const Vec3& axis, double angle) {
    double halfAngle = angle * 0.5;
    double s = std::sin(halfAngle);
    Vec3 n = normalized(axis);
    return {std::cos(halfAngle), n.x * s, n.y * s, n.z * s};
}

// --- Matrix Operations ---

inline Vec3 mat33MulVec3(const Mat3x3& m, const Vec3& v) {
    return {
        m.m[0] * v.x + m.m[1] * v.y + m.m[2] * v.z,
        m.m[3] * v.x + m.m[4] * v.y + m.m[5] * v.z,
        m.m[6] * v.x + m.m[7] * v.y + m.m[8] * v.z
    };
}

inline Mat3x3 mat33Mul(const Mat3x3& a, const Mat3x3& b) {
    Mat3x3 r;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r.at(i, j) = a.at(i, 0) * b.at(0, j) + a.at(i, 1) * b.at(1, j) + a.at(i, 2) * b.at(2, j);
        }
    }
    return r;
}

inline Mat3x3 mat33ExtractRotation(const Mat4x4& m) {
    return {
        (float)m.m[0], (float)m.m[1], (float)m.m[2],
        (float)m.m[4], (float)m.m[5], (float)m.m[6],
        (float)m.m[8], (float)m.m[9], (float)m.m[10]
    };
}

inline Mat3x3 qToMat33(const Quaternion& q) {
    double w2 = q.w * q.w, x2 = q.x * q.x, y2 = q.y * q.y, z2 = q.z * q.z;
    double xy = q.x * q.y, wz = q.w * q.z, xz = q.x * q.z, wy = q.w * q.y, yz = q.y * q.z, wx = q.w * q.x;
    return {
        w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), w2 - x2 + y2 - z2, 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), w2 - x2 - y2 + z2
    };
}

inline Mat3x3 skew(const Vec3& v) {
    return {
        0, -v.z, v.y,
        v.z, 0, -v.x,
        -v.y, v.x, 0
    };
}

inline Mat3x3 mat33Scale(const Mat3x3& a, double s) {
    Mat3x3 r;
    for (int i = 0; i < 9; i++) r.m[i] = a.m[i] * s;
    return r;
}

inline Mat3x3 mat33Add(const Mat3x3& a, const Mat3x3& b) {
    Mat3x3 r;
    for (int i = 0; i < 9; i++) r.m[i] = a.m[i] + b.m[i];
    return r;
}

inline Mat4x4 invert4x4_copy(const Mat4x4& in) {
    Mat4x4 out;
    invert4x4(in, out);
    return out;
}

inline Vec3 Sim3::transform(const Vec3& p) const {
    Mat3x3 R = qToMat33(q);
    Vec3 Rp = mat33MulVec3(R, p);
    return {s * Rp.x + t.x, s * Rp.y + t.y, s * Rp.z + t.z};
}

inline Mat3x3 mat33Transpose(const Mat3x3& a) {
    return {
        a.m[0], a.m[3], a.m[6],
        a.m[1], a.m[4], a.m[7],
        a.m[2], a.m[5], a.m[8]
    };
}

inline Quaternion qNormalize(const Quaternion& q) {
    double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n < 1e-12) return {1, 0, 0, 0};
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}

inline Quaternion mat33ToQuat(const Mat3x3& m) {
    double tr = m.m[0] + m.m[4] + m.m[8];
    Quaternion q;
    if (tr > 0) {
        double s = std::sqrt(tr + 1.0) * 2;
        q.w = 0.25 * s;
        q.x = (m.m[7] - m.m[5]) / s;
        q.y = (m.m[2] - m.m[6]) / s;
        q.z = (m.m[3] - m.m[1]) / s;
    } else if ((m.m[0] > m.m[4]) && (m.m[0] > m.m[8])) {
        double s = std::sqrt(1.0 + m.m[0] - m.m[4] - m.m[8]) * 2;
        q.w = (m.m[7] - m.m[5]) / s;
        q.x = 0.25 * s;
        q.y = (m.m[1] + m.m[3]) / s;
        q.z = (m.m[2] + m.m[6]) / s;
    } else if (m.m[4] > m.m[8]) {
        double s = std::sqrt(1.0 + m.m[4] - m.m[0] - m.m[8]) * 2;
        q.w = (m.m[2] - m.m[6]) / s;
        q.x = (m.m[1] + m.m[3]) / s;
        q.y = 0.25 * s;
        q.z = (m.m[5] + m.m[7]) / s;
    } else {
        double s = std::sqrt(1.0 + m.m[8] - m.m[0] - m.m[4]) * 2;
        q.w = (m.m[3] - m.m[1]) / s;
        q.x = (m.m[2] + m.m[6]) / s;
        q.y = (m.m[5] + m.m[7]) / s;
        q.z = 0.25 * s;
    }
    return qNormalize(q);
}

// --- Linear Solver (Gaussian Elimination) ---
template<int N>
inline bool solveLinear(double A[N*N], double B[N], double X[N]) {
    for (int i = 0; i < N; i++) {
        int pivot = i;
        for (int j = i + 1; j < N; j++) {
            if (std::abs(A[j * N + i]) > std::abs(A[pivot * N + i])) pivot = j;
        }
        for (int j = i; j < N; j++) std::swap(A[i * N + j], A[pivot * N + j]);
        std::swap(B[i], B[pivot]);

        if (std::abs(A[i * N + i]) < 1e-15) return false;

        for (int j = i + 1; j < N; j++) {
            double factor = A[j * N + i] / A[i * N + i];
            for (int k = i; k < N; k++) A[j * N + k] -= factor * A[i * N + k];
            B[j] -= factor * B[i];
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < N; j++) sum += A[i * N + j] * X[j];
        X[i] = (B[i] - sum) / A[i * N + i];
    }
    return true;
}

inline Vec4 solve4x4(const Mat4x4& A, const Vec4& B) {
    double mat[16], b[4], x[4];
    for (int i = 0; i < 16; i++) mat[i] = A.m[i];
    // Let's use a cleaner loop
    for (int i = 0; i < 4; i++) b[i] = B[i];
    if (!solveLinear<4>(mat, b, x)) return {0, 0, 0, 0};
    return {x[0], x[1], x[2], x[3]};
}

inline Vec3 solve3x3(const Mat3x3& A, const Vec3& B) {
    double mat[9], b[3], x[3];
    for (int i = 0; i < 9; i++) mat[i] = A.m[i];
    for (int i = 0; i < 3; i++) b[i] = B[i];
    if (!solveLinear<3>(mat, b, x)) return {0, 0, 0};
    return {x[0], x[1], x[2]};
}

} // namespace orb_lite

#endif
