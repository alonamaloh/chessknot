#include "core/board.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"
#include <iostream>

int main() {
  Board board;
  std::cout << board << "\n";

  search::Searcher searcher;
  searcher.set_verbose(true);

  // Search with a node limit
  auto tc = search::TimeControl::with_time(5.0, 10.0);
  auto result = searcher.search(board, 100, tc);

  std::cout << "\nBest move: " << result.best_move
            << " score: " << result.score
            << " depth: " << result.depth
            << " nodes: " << result.nodes << "\n";

  return 0;
}
