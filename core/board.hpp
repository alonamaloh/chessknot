#pragma once

#include <cstdint>
#include <ostream>
#include <utility>

using Bb = std::uint64_t;

constexpr Bb FILE_A = 0x0101010101010101ULL;
constexpr Bb FILE_H = 0x8080808080808080ULL;
constexpr Bb RANK_1 = 0x00000000000000FFULL;
constexpr Bb RANK_8 = 0xFF00000000000000ULL;

// Orthogonal neighbors of all set bits
inline Bb neighbors(Bb bb) {
  return (bb << 8) | (bb >> 8) |
         ((bb & ~FILE_H) << 1) | ((bb & ~FILE_A) >> 1);
}

// A move is (from_square, to_square), each 0..63
struct Move {
  std::uint8_t from;
  std::uint8_t to;

  Move() : from(0), to(0) {}
  Move(int f, int t) : from(f), to(t) {}

  bool operator==(const Move& o) const { return from == o.from && to == o.to; }
  bool operator<(const Move& o) const {
    if (from != o.from) return from < o.from;
    return to < o.to;
  }
};

// Fixed-size vector to avoid heap allocations in hot paths
template<typename T, int MaxSize>
struct FixedVector {
  T data[MaxSize];
  std::size_t count = 0;

  T const& operator[](std::size_t i) const { return data[i]; }
  T& operator[](std::size_t i) { return data[i]; }

  void push_back(const T& val) { data[count++] = val; }
  void clear() { count = 0; }
  bool empty() const { return count == 0; }
  std::size_t size() const { return count; }

  T* begin() { return data; }
  T* end() { return data + count; }
  const T* begin() const { return data; }
  const T* end() const { return data + count; }
};

// 16 pieces * 32 empty squares = 512 max moves
using MoveList = FixedVector<Move, 512>;

// Board state: white is always the side to move.
// After each move we swap white and black.
struct Board {
  Bb white;
  Bb black;

  // Initial position: white on white squares in odd ranks, black in even ranks
  static constexpr Bb INIT_WHITE = 0x00AA00AA00AA00AAULL;
  static constexpr Bb INIT_BLACK = 0x5500550055005500ULL;

  Board() : white(INIT_WHITE), black(INIT_BLACK) {}
  Board(Bb w, Bb b) : white(w), black(b) {}

  Bb occupied() const { return white | black; }
  Bb empty() const { return ~occupied(); }

  std::uint64_t hash() const {
    std::uint64_t h = white * 0x9d82c4a44a2de231ULL;
    h ^= h >> 32;
    h += black;
    h *= 0xb20534a511d28c31ULL;
    h ^= h >> 32;
    return h;
  }
};

inline bool operator==(const Board& a, const Board& b) {
  return a.white == b.white && a.black == b.black;
}

namespace std {
  template<>
  struct hash<Board> {
    std::size_t operator()(const Board& b) const noexcept {
      return b.hash();
    }
  };
}

// Apply a move and swap sides (so white is always to move)
inline Board makeMove(Board board, Move move) {
  board.white ^= (1ULL << move.from) | (1ULL << move.to);
  std::swap(board.white, board.black);
  return board;
}

std::ostream& operator<<(std::ostream& os, const Board& board);
std::ostream& operator<<(std::ostream& os, const Move& move);
