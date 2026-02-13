#include "core/board.hpp"
#include "core/movegen.hpp"
#include <chrono>
#include <iostream>

int main() {
  Board board;
  std::cout << board << "\n";

  MoveList moves;
  generateMoves(board, moves);
  std::cout << "Legal moves from initial position: " << moves.size() << "\n\n";

  for (int depth = 0; depth <= 3; ++depth) {
    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t nodes = perft(board, depth);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "perft(" << depth << ") = " << nodes;
    if (secs > 0.001)
      std::cout << "  (" << secs << "s, "
                << (std::uint64_t)(nodes / secs) << " nps)";
    std::cout << "\n";
  }
  return 0;
}
