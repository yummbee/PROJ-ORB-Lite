#include "Map.hpp"
#include <fstream>
#include <cmath>
#include <iostream>

namespace orb_lite {

int Map::addPoint(const Vec3& pos, const Descriptor& desc, int refKfId) {
    int id = (int)points.size();
    MapPoint mp = {pos, desc, 1, 1, refKfId};
    mp.pMap = this;
    points.push_back(mp);
    return id;
}

int Map::addKeyframe(KeyFrame& kf) {
    int id = (int)keyframes.size();
    kf.pMap = this;
    keyframes.push_back(kf);
    return id;
}

void Map::save(const char* filename) const {
    std::ofstream f(filename, std::ios::binary);
    size_t numPoints = points.size();
    f.write((char*)&numPoints, sizeof(size_t));
    for (const auto& p : points) {
        f.write((const char*)&p, sizeof(MapPoint));
    }
}

void Map::load(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    size_t numPoints;
    f.read((char*)&numPoints, sizeof(size_t));
    points.resize(numPoints);
    for (size_t i = 0; i < numPoints; i++) {
        f.read((char*)&points[i], sizeof(MapPoint));
    }
}

void Map::saveCSV(const char* filename) const {
    std::ofstream f(filename);
    f << "type,x,y,z,nFound,nVisible" << std::endl;
    int exported_count = 0;
    for (const auto& p : points) {
        // --- RULE 1: No Bad Points ---
        if (p.isBad) continue;

        // --- RULE 2: The Structural Persistence Test ---
        if (p.nFound < 15) continue;

        f << "point," << p.pos.x << "," << p.pos.y << "," << p.pos.z << "," 
          << p.nFound << "," << p.nVisible << std::endl;
        exported_count++;
    }
    for (const auto& kf : keyframes) {
        f << "kf," << kf.pose.m[3] << "," << kf.pose.m[7] << "," << kf.pose.m[11] << ",0,0" << std::endl;
    }
    f.close();
    std::cout << "-> Saved " << filename << " (" << exported_count 
              << " structural points exported. " 
              << (points.size() - exported_count) << " noise points filtered.)\n";
}

bool triangulate(const Vec3& o1, const Vec3& d1, const Vec3& o2, const Vec3& d2, Vec3& p) {
    Vec3 w0 = {o1.x - o2.x, o1.y - o2.y, o1.z - o2.z};
    double a = dot(d1, d1);
    double b = dot(d1, d2);
    double c = dot(d2, d2);
    double d = dot(d1, w0);
    double e = dot(d2, w0);
    double denom = a * c - b * b;
    if (std::abs(denom) < 1e-9) return false;
    double t1 = (b * e - c * d) / denom;
    double t2 = (a * e - b * d) / denom;
    Vec3 p1 = {o1.x + t1 * d1.x, o1.y + t1 * d1.y, o1.z + t1 * d1.z};
    Vec3 p2 = {o2.x + t2 * d2.x, o2.y + t2 * d2.y, o2.z + t2 * d2.z};
    p = {(p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5, (p1.z + p2.z) * 0.5};
    if (t1 <= 0 || t2 <= 0) return false; 
    return true;
}

void Map::mergePoints(int id1, int id2) {
    if (id1 == id2) return;
    for (auto& kf : keyframes) {
        for (int& mp_id : kf.mapPointIds) {
            if (mp_id == id2) mp_id = id1;
        }
    }
    points[id2].isBad = true;
}

void Map::cleanMap() {
    // We don't actually remove from deque as it breaks indices,
    // but we could compact it later if needed.
}

} // namespace orb_lite
