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
  w1_ = readFloats(in, 128 * 45);
  b1_ = readFloats(in, 128);
  w2_ = readFloats(in, 64 * 128);
  b2_ = readFloats(in, 64);
  w3_ = readFloats(in, 1 * 64);
  b3_ = readFloats(in, 1);
}

int MLP::evaluate(const Board& board) const {
  // Extract features
  std::uint8_t counts[NUM_FEATURES];
  computeClassCounts(board, counts);

  // Normalize to [0, 1]
  float input[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; ++i)
    input[i] = counts[i] / 64.0f;

  // Layer 1: 45 -> 128, ReLU
  float h1[128];
  for (int i = 0; i < 128; ++i) {
    float sum = b1_[i];
    for (int j = 0; j < 45; ++j)
      sum += w1_[i * 45 + j] * input[j];
    h1[i] = std::max(0.0f, sum);
  }

  // Layer 2: 128 -> 64, ReLU
  float h2[64];
  for (int i = 0; i < 64; ++i) {
    float sum = b2_[i];
    for (int j = 0; j < 128; ++j)
      sum += w2_[i * 128 + j] * h1[j];
    h2[i] = std::max(0.0f, sum);
  }

  // Layer 3: 64 -> 1, tanh
  float sum = b3_[0];
  for (int j = 0; j < 64; ++j)
    sum += w3_[j] * h2[j];
  float out = std::tanh(sum);

  // Scale to [-10000, +10000]
  return static_cast<int>(out * 10000.0f);
}

} // namespace nn
