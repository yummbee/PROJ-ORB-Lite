#ifndef ORB_LITE_BOW_HPP
#define ORB_LITE_BOW_HPP

#include "Vision.hpp"
#include <vector>
#include <string>
#include <stdint.h>

namespace orb_lite {

typedef uint32_t WordId;
typedef uint32_t NodeId;
typedef double WordValue;

// BoW Vector: Sorted vector of WordId -> Weight
class BowVector : public std::vector<std::pair<WordId, WordValue>> {
public:
    void addWeight(WordId id, WordValue v);
    void normalize(); // L1 normalization
};

// Feature Vector: Sorted vector of NodeId -> Indices
class FeatureVector : public std::vector<std::pair<NodeId, std::vector<unsigned int>>> {
public:
    void addFeature(NodeId id, unsigned int i);
};

enum WeightingType {
    TF_IDF,
    TF,
    IDF,
    BINARY
};

enum ScoringType {
    L1_NORM,
    L2_NORM,
    CHI_SQUARE,
    KL,
    BHATTACHARYYA,
    DOT_PRODUCT
};

class Vocabulary {
public:
    struct Node {
        NodeId id;
        WordId word_id;
        WordValue weight;
        NodeId parent;
        std::vector<NodeId> children;
        Descriptor descriptor;

        Node() : id(0), word_id(0), weight(0), parent(0) {}
    };

    Vocabulary();
    Vocabulary(const std::string& filename);
    ~Vocabulary();

    bool loadFromTextFile(const std::string& filename);
    
    // Transform features to BowVector and FeatureVector
    void transform(const std::vector<Descriptor>& features, BowVector& v, FeatureVector& fv, int levels_up) const;
    
    double score(const BowVector& v1, const BowVector& v2) const;

    size_t size() const { return m_nodes.size(); }
    int getDepthLevels() const { return m_L; }
    int getBranchingFactor() const { return m_k; }

protected:
    void transform(const Descriptor& feature, WordId& id, WordValue& weight, NodeId* nid, int levelsup) const;
    static double distance(const Descriptor& a, const Descriptor& b);

    int m_k; // branching factor
    int m_L; // depth levels
    WeightingType m_weighting;
    ScoringType m_scoring;

    std::vector<Node> m_nodes;
    std::vector<Node*> m_words; // leaves
};

} // namespace orb_lite

#endif
