#include "board.hpp"

static char squareName(int sq, char* buf) {
  buf[0] = 'a' + (sq & 7);
  buf[1] = '1' + (sq >> 3);
  return 0;
}

std::ostream& operator<<(std::ostream& os, const Move& move) {
  char buf[3] = {};
  squareName(move.from, buf);
  os << buf << '-';
  squareName(move.to, buf);
  os << buf;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Board& board) {
  // Print from rank 8 (top) to rank 1 (bottom)
  for (int rank = 7; rank >= 0; --rank) {
    os << (rank + 1) << ' ';
    for (int file = 0; file < 8; ++file) {
      int sq = rank * 8 + file;
      Bb bit = 1ULL << sq;
      if (board.white & bit)
        os << " W";
      else if (board.black & bit)
        os << " B";
      else
        os << " .";
    }
    os << '\n';
  }
  os << "  ";
  for (int file = 0; file < 8; ++file)
    os << ' ' << (char)('a' + file);
  os << '\n';
  return os;
}
