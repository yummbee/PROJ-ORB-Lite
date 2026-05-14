#include "BoW.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace orb_lite {

void BowVector::addWeight(WordId id, WordValue v) {
    auto it = std::lower_bound(begin(), end(), id, [](const std::pair<WordId, WordValue>& p, WordId id) {
        return p.first < id;
    });
    if(it != end() && it->first == id) {
        it->second += v;
    } else {
        insert(it, std::make_pair(id, v));
    }
}

void BowVector::normalize() {
    double norm = 0;
    for(auto it = begin(); it != end(); ++it) {
        norm += std::abs(it->second);
    }
    if(norm > 0) {
        for(auto it = begin(); it != end(); ++it) {
            it->second /= norm;
        }
    }
}

void FeatureVector::addFeature(NodeId id, unsigned int i) {
    auto it = std::lower_bound(begin(), end(), id, [](const std::pair<NodeId, std::vector<unsigned int>>& p, NodeId id) {
        return p.first < id;
    });
    if(it != end() && it->first == id) {
        it->second.push_back(i);
    } else {
        insert(it, std::make_pair(id, std::vector<unsigned int>{i}));
    }
}

Vocabulary::Vocabulary() : m_k(10), m_L(5), m_weighting(TF_IDF), m_scoring(L1_NORM) {}

Vocabulary::Vocabulary(const std::string& filename) {
    loadFromTextFile(filename);
}

Vocabulary::~Vocabulary() {}

bool Vocabulary::loadFromTextFile(const std::string& filename) {
    std::ifstream f(filename.c_str());
    if(!f.is_open()) return false;

    std::string line;
    std::getline(f, line);
    if(line.empty()) return false;
    
    std::stringstream ss(line);
    int k, L, weighting, scoring;
    ss >> k >> L >> weighting >> scoring;
    
    m_k = k;
    m_L = L;
    m_weighting = (WeightingType)weighting;
    m_scoring = (ScoringType)scoring;

    m_nodes.clear();
    m_words.clear();

    // First pass: Load all nodes
    while(std::getline(f, line)) {
        if(line.empty()) continue;
        std::stringstream ss2(line);
        int id, parent;
        ss2 >> id >> parent;
        
        if(id >= (int)m_nodes.size()) m_nodes.resize(id + 1);
        
        Node& n = m_nodes[id];
        n.id = id;
        n.parent = parent;
        
        // Descriptor: 32 bytes (uint8_t)
        uint8_t* p_desc = (uint8_t*)n.descriptor.val;
        for(int i=0; i<32; i++) {
            int v;
            if(!(ss2 >> v)) break;
            p_desc[i] = (uint8_t)v;
        }
        
        if(!(ss2 >> n.weight)) n.weight = 0;
        
        if(id != 0 && parent >= 0 && parent < (int)m_nodes.size()) {
            m_nodes[parent].children.push_back(id);
        }


    }
    
    // Second pass: Identify words (leaves)
    for (size_t i = 0; i < m_nodes.size(); i++) {
        if (m_nodes[i].children.empty()) {
            m_nodes[i].word_id = m_words.size();
            m_words.push_back(&m_nodes[i]);
        }
    }
    
    std::cout << "Vocabulary loaded: " << m_nodes.size() << " nodes, " << m_words.size() << " words." << std::endl;
    return true;
}

bool Vocabulary::saveToBinaryFile(const std::string& filename) {
    std::ofstream f(filename.c_str(), std::ios::binary);
    if(!f.is_open()) return false;

    // Header
    uint32_t k = (uint32_t)m_k;
    uint32_t L = (uint32_t)m_L;
    uint32_t weighting = (uint32_t)m_weighting;
    uint32_t scoring = (uint32_t)m_scoring;
    uint32_t num_nodes = (uint32_t)m_nodes.size();
    
    f.write((char*)&k, 4);
    f.write((char*)&L, 4);
    f.write((char*)&weighting, 4);
    f.write((char*)&scoring, 4);
    f.write((char*)&num_nodes, 4);

    for(const auto& n : m_nodes) {
        f.write((char*)&n.id, sizeof(NodeId));
        f.write((char*)&n.parent, sizeof(NodeId));
        f.write((char*)&n.weight, sizeof(WordValue));
        
        uint32_t num_children = (uint32_t)n.children.size();
        f.write((char*)&num_children, 4);
        for(NodeId cid : n.children) {
            f.write((char*)&cid, sizeof(NodeId));
        }
        
        f.write((char*)n.descriptor.val, 32);
    }
    
    std::cout << "Vocabulary saved to binary: " << filename << std::endl;
    return true;
}

bool Vocabulary::loadFromBinaryFile(const std::string& filename) {
    std::ifstream f(filename.c_str(), std::ios::binary);
    if(!f.is_open()) return false;

    uint32_t k, L, weighting, scoring, num_nodes;
    f.read((char*)&k, 4);
    f.read((char*)&L, 4);
    f.read((char*)&weighting, 4);
    f.read((char*)&scoring, 4);
    f.read((char*)&num_nodes, 4);

    m_k = k;
    m_L = L;
    m_weighting = (WeightingType)weighting;
    m_scoring = (ScoringType)scoring;

    m_nodes.clear();
    m_nodes.resize(num_nodes);
    m_words.clear();

    for(uint32_t i=0; i<num_nodes; i++) {
        Node& n = m_nodes[i];
        f.read((char*)&n.id, sizeof(NodeId));
        f.read((char*)&n.parent, sizeof(NodeId));
        f.read((char*)&n.weight, sizeof(WordValue));
        
        uint32_t num_children;
        f.read((char*)&num_children, 4);
        n.children.resize(num_children);
        for(uint32_t j=0; j<num_children; j++) {
            f.read((char*)&n.children[j], sizeof(NodeId));
        }
        
        f.read((char*)n.descriptor.val, 32);
    }
    
    // Identify words
    for (size_t i = 0; i < m_nodes.size(); i++) {
        if (m_nodes[i].children.empty() && m_nodes[i].weight > 0) {
            m_nodes[i].word_id = m_words.size();
            m_words.push_back(&m_nodes[i]);
        }
    }

    std::cout << "Vocabulary loaded (binary): " << m_nodes.size() << " nodes, " << m_words.size() << " words." << std::endl;
    return true;
}

void Vocabulary::transform(const std::vector<Descriptor>& features, BowVector& v, FeatureVector& fv, int levels_up) const {
    v.clear();
    fv.clear();
    
    if(m_nodes.empty()) return;

    for(size_t i=0; i<features.size(); i++) {
        WordId wid;
        WordValue weight;
        NodeId nid;
        transform(features[i], wid, weight, &nid, levels_up);
        
        if(weight > 0) {
            v.addWeight(wid, weight);
            fv.addFeature(nid, (unsigned int)i);
        }
    }
    
    v.normalize();
}

void Vocabulary::transform(const Descriptor& feature, WordId& id, WordValue& weight, NodeId* nid, int levelsup) const {
    NodeId current_node = 0; 
    int current_level = 0;
    
    while(!m_nodes[current_node].children.empty()) {
        const std::vector<NodeId>& children = m_nodes[current_node].children;
        double best_dist = distance(feature, m_nodes[children[0]].descriptor);
        NodeId best_child = children[0];
        
        for(size_t i=1; i<children.size(); i++) {
            double d = distance(feature, m_nodes[children[i]].descriptor);
            if(d < best_dist) {
                best_dist = d;
                best_child = children[i];
            }
        }
        
        current_node = best_child;
        current_level++;
        
        if(levelsup > 0 && m_L - current_level == levelsup) {
            if(nid) *nid = current_node;
        }
    }
    
    if(nid && (levelsup <= 0 || m_L - current_level < levelsup)) *nid = current_node;
    
    id = m_nodes[current_node].word_id;
    weight = m_nodes[current_node].weight;
}

double Vocabulary::distance(const Descriptor& a, const Descriptor& b) {
    int dist = 0;
    for(int i=0; i<8; i++) {
        dist += __builtin_popcount(a.val[i] ^ b.val[i]);
    }
    return (double)dist;
}

double Vocabulary::score(const BowVector& v1, const BowVector& v2) const {
    double s = 0;
    auto it1 = v1.begin();
    auto it2 = v2.begin();
    
    while(it1 != v1.end() && it2 != v2.end()) {
        if(it1->first == it2->first) {
            s += std::abs(it1->second) + std::abs(it2->second) - std::abs(it1->second - it2->second);
            it1++;
            it2++;
        } else if(it1->first < it2->first) {
            it1++;
        } else {
            it2++;
        }
    }
    return s / 2.0;
}

} // namespace orb_lite
