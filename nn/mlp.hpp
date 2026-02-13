#pragma once

#include "../core/board.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace nn {

// Simple MLP: 45 -> 32 (ReLU) -> 1 (tanh)
// Output scaled to [-10000, +10000] for use as search eval.
class MLP {
public:
  explicit MLP(const std::string& path);

  // Evaluate a position from the side to move's perspective.
  // Returns a score in [-10000, +10000].
  int evaluate(const Board& board) const;

private:
  // Weights stored row-major: w1[32][45], w2[1][32]
  std::vector<float> w1_, b1_;  // 32x45, 32
  std::vector<float> w2_, b2_;  // 1x32, 1
};

} // namespace nn
