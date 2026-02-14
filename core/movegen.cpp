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

  // enemyCount[sq] encoded in bits: H=bit2, M=bit1, L=bit0
  // We need to compare source vs target counts.
  // Pack per-square count into a lookup via the bitboards.

  Bb white = board.white;

  // For each white piece, try moves
  Bb pieces = white;
  while (pieces) {
    int sq = __builtin_ctzll(pieces);
    pieces &= pieces - 1;

    // Source enemy neighbor count
    int srcCount = ((H >> sq) & 1) * 4 + ((M >> sq) & 1) * 2 + ((L >> sq) & 1);

#ifdef ROOK_MOVES
    // Rook sliding: cast rays in 4 orthogonal directions
    int file = sq & 7;
    int rank = sq >> 3;

    // Up (+8)
    for (int r = rank + 1; r < 8; ++r) {
      int tsq = r * 8 + file;
      if ((1ULL << tsq) & board.occupied()) break;
      int dstCount = ((H >> tsq) & 1) * 4 + ((M >> tsq) & 1) * 2 + ((L >> tsq) & 1);
      if (dstCount >= srcCount)
        moves.push_back(Move(sq, tsq));
    }
    // Down (-8)
    for (int r = rank - 1; r >= 0; --r) {
      int tsq = r * 8 + file;
      if ((1ULL << tsq) & board.occupied()) break;
      int dstCount = ((H >> tsq) & 1) * 4 + ((M >> tsq) & 1) * 2 + ((L >> tsq) & 1);
      if (dstCount >= srcCount)
        moves.push_back(Move(sq, tsq));
    }
    // Right (+1)
    for (int f = file + 1; f < 8; ++f) {
      int tsq = rank * 8 + f;
      if ((1ULL << tsq) & board.occupied()) break;
      int dstCount = ((H >> tsq) & 1) * 4 + ((M >> tsq) & 1) * 2 + ((L >> tsq) & 1);
      if (dstCount >= srcCount)
        moves.push_back(Move(sq, tsq));
    }
    // Left (-1)
    for (int f = file - 1; f >= 0; --f) {
      int tsq = rank * 8 + f;
      if ((1ULL << tsq) & board.occupied()) break;
      int dstCount = ((H >> tsq) & 1) * 4 + ((M >> tsq) & 1) * 2 + ((L >> tsq) & 1);
      if (dstCount >= srcCount)
        moves.push_back(Move(sq, tsq));
    }
#else
    // Wazir: try 4 orthogonal neighbors
    Bb empty = board.empty();
    int file = sq & 7;
    int targets[4];
    int ntargets = 0;

    if (sq + 8 < 64) targets[ntargets++] = sq + 8;  // up
    if (sq - 8 >= 0)  targets[ntargets++] = sq - 8;  // down
    if (file < 7)      targets[ntargets++] = sq + 1;  // right
    if (file > 0)      targets[ntargets++] = sq - 1;  // left

    for (int i = 0; i < ntargets; ++i) {
      int tsq = targets[i];
      Bb tBit = 1ULL << tsq;
      if (!(tBit & empty)) continue;

      int dstCount = ((H >> tsq) & 1) * 4 + ((M >> tsq) & 1) * 2 + ((L >> tsq) & 1);
      if (dstCount >= srcCount)
        moves.push_back(Move(sq, tsq));
    }
#endif
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
