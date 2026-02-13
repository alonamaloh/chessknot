#pragma once

#include "board.hpp"
#include "features.hpp"
#include <cstdio>
#include <cmath>

constexpr int NUM_MOVE_CLASSES = 225;  // 15 before-configs x 15 after-configs

// Compute the move class (0-224) for a move on the given board.
// Based on (white_neighbors, black_neighbors) of the piece at its from-square
// (before the move) and at its to-square (after the move, before side swap).
inline int moveClass(const Board& board, Move move) {
  Bb from_bit = 1ULL << move.from;
  Bb to_bit = 1ULL << move.to;

  // Neighbor counts at 'from' on the current board
  Bb from_nbrs = neighbors(from_bit);
  int w_before = __builtin_popcountll(from_nbrs & board.white);
  int b_before = __builtin_popcountll(from_nbrs & board.black);

  // Board after moving the piece (before side swap)
  Bb modified_white = board.white ^ (from_bit | to_bit);

  // Neighbor counts at 'to' on the modified board
  Bb to_nbrs = neighbors(to_bit);
  int w_after = __builtin_popcountll(to_nbrs & modified_white);
  int b_after = __builtin_popcountll(to_nbrs & board.black);

  int before_idx = NEIGHBOR_OFFSET[w_before] + b_before;
  int after_idx = NEIGHBOR_OFFSET[w_after] + b_after;
  return before_idx * 15 + after_idx;
}

struct PolicyTable {
  float logits[NUM_MOVE_CLASSES] = {};  // zero-initialized (uniform)

  bool save(const char* path) const {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fwrite(logits, sizeof(float), NUM_MOVE_CLASSES, f);
    std::fclose(f);
    return true;
  }

  bool load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    bool ok = std::fread(logits, sizeof(float), NUM_MOVE_CLASSES, f) == NUM_MOVE_CLASSES;
    std::fclose(f);
    return ok;
  }
};
