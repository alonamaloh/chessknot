#include "movegen.hpp"

std::size_t generateMoves(const Board& board, MoveList& moves) {
  moves.clear();

  // Classify all squares by enemy neighbor count using a full adder
  Bb enemy = board.black;
  Bb a = enemy << 8;
  Bb b = enemy >> 8;
  Bb c = (enemy & ~FILE_H) << 1;
  Bb d = (enemy & ~FILE_A) >> 1;

  Bb ab_xor = a ^ b;
  Bb ab_and = a & b;
  Bb L_abc = ab_xor ^ c;
  Bb M_abc = ab_and | (ab_xor & c);
  Bb L = L_abc ^ d;
  Bb carry = L_abc & d;
  Bb M = M_abc ^ carry;
  Bb H = M_abc & carry;

  Bb exactly[5] = {
    ~(L | M | H),
    L & ~M,
    M & ~L,
    M & L,
    H
  };

  Bb empty = board.empty();
  Bb white = board.white;

  // Generate in order of increasing delta (to - from):
  // try moves that don't increase neighbor count first
  for (int delta = 0; delta <= 4; ++delta) {
    for (int from = 0; from + delta <= 4; ++from) {
      int to = from + delta;
      Bb sources = exactly[from] & white;
      Bb targets = exactly[to] & empty;
      while (sources) {
        int fsq = __builtin_ctzll(sources);
        sources &= sources - 1;
        Bb t = targets;
        while (t) {
          int tsq = __builtin_ctzll(t);
          t &= t - 1;
          moves.push_back(Move(fsq, tsq));
        }
      }
    }
  }

  return moves.size();
}

std::uint64_t perft(const Board& board, int depth) {
  if (depth == 0) return 1;

  MoveList moves;
  generateMoves(board, moves);

  if (depth == 1) return moves.size();

  std::uint64_t count = 0;
  for (const auto& move : moves) {
    count += perft(makeMove(board, move), depth - 1);
  }
  return count;
}
