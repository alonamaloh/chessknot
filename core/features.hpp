#pragma once

#include "board.hpp"
#include <cstring>

// Number of square classes:
//   3 contents (empty=0, white=1, black=2) x
//   15 neighbor configs (w_neighbors, b_neighbors) with w+b <= 4
constexpr int NUM_FEATURES = 45;

// Neighbor config index: offset[w] + b, where w+b <= 4
constexpr int NEIGHBOR_OFFSET[5] = {0, 5, 9, 12, 14};

// Classify all 64 squares and count occurrences of each class.
inline void computeClassCounts(const Board& board, std::uint8_t counts[NUM_FEATURES]) {
  // Full adder for white neighbor counts
  Bb w = board.white;
  Bb aw = w << 8, bw = w >> 8;
  Bb cw = (w & ~FILE_H) << 1, dw = (w & ~FILE_A) >> 1;
  Bb ab_xor = aw ^ bw, ab_and = aw & bw;
  Bb Labc = ab_xor ^ cw, Mabc = ab_and | (ab_xor & cw);
  Bb Lw = Labc ^ dw, carry = Labc & dw;
  Bb Mw = Mabc ^ carry, Hw = Mabc & carry;
  Bb ew[5] = {~(Lw|Mw|Hw), Lw & ~Mw, Mw & ~Lw, Mw & Lw, Hw};

  // Full adder for black neighbor counts
  Bb b = board.black;
  Bb ab2 = b << 8, bb2 = b >> 8;
  Bb cb2 = (b & ~FILE_H) << 1, db2 = (b & ~FILE_A) >> 1;
  ab_xor = ab2 ^ bb2; ab_and = ab2 & bb2;
  Labc = ab_xor ^ cb2; Mabc = ab_and | (ab_xor & cb2);
  Bb Lb = Labc ^ db2; carry = Labc & db2;
  Bb Mb = Mabc ^ carry, Hb = Mabc & carry;
  Bb eb[5] = {~(Lb|Mb|Hb), Lb & ~Mb, Mb & ~Lb, Mb & Lb, Hb};

  Bb content[3] = {board.empty(), board.white, board.black};

  std::memset(counts, 0, NUM_FEATURES);
  for (int c = 0; c < 3; ++c)
    for (int wn = 0; wn <= 4; ++wn)
      for (int bn = 0; wn + bn <= 4; ++bn)
        counts[c * 15 + NEIGHBOR_OFFSET[wn] + bn] =
            static_cast<std::uint8_t>(__builtin_popcountll(content[c] & ew[wn] & eb[bn]));
}
