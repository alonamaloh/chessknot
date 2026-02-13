#pragma once

#include "../core/board.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace nn {

// Simple MLP: 45 -> 128 (ReLU) -> 64 (ReLU) -> 1 (tanh)
// Output scaled to [-10000, +10000] for use as search eval.
class MLP {
public:
  explicit MLP(const std::string& path);

  // Evaluate a position from the side to move's perspective.
  // Returns a score in [-10000, +10000].
  int evaluate(const Board& board) const;

private:
  // Weights stored row-major: w1[128][45], w2[64][128], w3[1][64]
  std::vector<float> w1_, b1_;  // 128x45, 128
  std::vector<float> w2_, b2_;  // 64x128, 64
  std::vector<float> w3_, b3_;  // 1x64, 1
};

} // namespace nn
