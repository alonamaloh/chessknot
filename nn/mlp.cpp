#include "mlp.hpp"
#include "../core/features.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace nn {

static std::vector<float> readFloats(std::ifstream& in, std::size_t count) {
  std::vector<float> v(count);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  if (!in) throw std::runtime_error("unexpected end of model file");
  return v;
}

MLP::MLP(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open model: " + path);

  // Read layers in order: weights then biases
  w1_ = readFloats(in, 32 * 45);
  b1_ = readFloats(in, 32);
  w2_ = readFloats(in, 1 * 32);
  b2_ = readFloats(in, 1);
}

int MLP::evaluate(const Board& board) const {
  // Extract features
  std::uint8_t counts[NUM_FEATURES];
  computeClassCounts(board, counts);

  // Normalize to [0, 1]
  float input[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; ++i)
    input[i] = counts[i] / 64.0f;

  // Layer 1: 45 -> 32, ReLU
  float h1[32];
  for (int i = 0; i < 32; ++i) {
    float sum = b1_[i];
    for (int j = 0; j < 45; ++j)
      sum += w1_[i * 45 + j] * input[j];
    h1[i] = std::max(0.0f, sum);
  }

  // Layer 2: 32 -> 1, tanh
  float sum = b2_[0];
  for (int j = 0; j < 32; ++j)
    sum += w2_[j] * h1[j];
  float out = std::tanh(sum);

  // Scale to [-10000, +10000]
  return static_cast<int>(out * 10000.0f);
}

} // namespace nn
