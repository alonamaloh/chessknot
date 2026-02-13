#include "core/board.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"
#include <H5Cpp.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Classify all 64 squares into 45 classes:
//   3 contents (empty=0, white=1, black=2) x
//   15 neighbor configs (w_neighbors, b_neighbors) with w+b <= 4
void computeClassCounts(const Board& board, std::uint8_t counts[45]) {
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
  constexpr int offset[5] = {0, 5, 9, 12, 14};

  std::memset(counts, 0, 45);
  for (int c = 0; c < 3; ++c)
    for (int wn = 0; wn <= 4; ++wn)
      for (int bn = 0; wn + bn <= 4; ++bn)
        counts[c * 15 + offset[wn] + bn] =
            static_cast<std::uint8_t>(__builtin_popcountll(content[c] & ew[wn] & eb[bn]));
}

// Play one self-play game. Returns: +1 first mover wins, -1 second mover wins, 0 draw.
int playGame(search::Searcher& searcher, std::uint64_t seed,
             std::vector<std::uint64_t>& all_white,
             std::vector<std::uint64_t>& all_black,
             std::vector<std::uint8_t>& all_counts,  // flattened, 45 per position
             std::vector<std::int8_t>& all_outcome,
             int node_limit, int max_plies) {
  Board board;

  auto eval = [seed](const Board& b) -> int {
    std::uint64_t h = b.hash() ^ seed;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int>(static_cast<std::int64_t>(h) % 10001);
  };
  searcher.set_eval(eval);
  searcher.clear_tt();

  std::vector<std::uint64_t> hashes;
  std::size_t start = all_outcome.size();
  int game_result = 0;

  for (int ply = 0; ply < max_plies; ++ply) {
    std::uint64_t h = board.hash();

    // 3-fold repetition check
    int rep_count = 0;
    for (auto ph : hashes)
      if (ph == h) rep_count++;
    if (rep_count >= 2) break;  // draw
    hashes.push_back(h);

    MoveList moves;
    generateMoves(board, moves);
    if (moves.empty()) {
      game_result = (ply % 2 == 0) ? -1 : 1;
      break;
    }

    // Record position features
    all_white.push_back(board.white);
    all_black.push_back(board.black);

    std::uint8_t counts[45];
    computeClassCounts(board, counts);
    all_counts.insert(all_counts.end(), counts, counts + 45);

    all_outcome.push_back(0);  // placeholder

    // Set game history for repetition detection in search
    std::vector<std::uint64_t> history;
    for (int i = static_cast<int>(hashes.size()) - 2; i >= 0; --i)
      history.push_back(hashes[i]);
    searcher.set_history(history);

    auto tc = search::TimeControl::with_nodes(node_limit);
    auto result = searcher.search(board, 100, tc);
    board = makeMove(board, result.best_move);
  }

  // Label all positions from this game
  for (std::size_t i = start; i < all_outcome.size(); ++i) {
    std::size_t ply = i - start;
    if (game_result == 0)
      all_outcome[i] = 0;
    else
      all_outcome[i] = static_cast<std::int8_t>(
          (ply % 2 == 0) ? game_result : -game_result);
  }

  return game_result;
}

int main(int argc, char* argv[]) {
  int num_games = 1000;
  int node_limit = 10000;
  const char* output_file = "training_data.h5";

  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::cout << "Usage: selfplay [num_games] [nodes_per_move] [output_file]\n"
              << "  num_games      Number of self-play games (default: 1000)\n"
              << "  nodes_per_move Node limit per search (default: 10000)\n"
              << "  output_file    HDF5 output path (default: training_data.h5)\n";
    return 0;
  }

  if (argc > 1) num_games = std::atoi(argv[1]);
  if (argc > 2) node_limit = std::atoi(argv[2]);
  if (argc > 3) output_file = argv[3];

  std::cout << "Self-play: " << num_games << " games, "
            << node_limit << " nodes/move, output: " << output_file << std::endl;

  std::mt19937_64 rng(42);
  search::Searcher searcher;

  std::vector<std::uint64_t> all_white, all_black;
  std::vector<std::uint8_t> all_counts;
  std::vector<std::int8_t> all_outcome;

  int wins = 0, losses = 0, draws = 0;

  for (int game = 0; game < num_games; ++game) {
    std::uint64_t seed = rng();
    int result = playGame(searcher, seed, all_white, all_black,
                          all_counts, all_outcome, node_limit, 500);

    if (result > 0) wins++;
    else if (result < 0) losses++;
    else draws++;

    if ((game + 1) % 100 == 0 || game + 1 == num_games)
      std::cout << "Game " << (game + 1) << "/" << num_games
                << "  W:" << wins << " B:" << losses << " D:" << draws
                << "  positions:" << all_outcome.size() << std::endl;
  }

  // Write HDF5
  std::size_t n = all_outcome.size();
  {
    H5::H5File file(output_file, H5F_ACC_TRUNC);

    hsize_t dims1[1] = {n};
    H5::DataSpace space1(1, dims1);

    file.createDataSet("white", H5::PredType::NATIVE_UINT64, space1)
        .write(all_white.data(), H5::PredType::NATIVE_UINT64);

    file.createDataSet("black", H5::PredType::NATIVE_UINT64, space1)
        .write(all_black.data(), H5::PredType::NATIVE_UINT64);

    hsize_t dims2[2] = {n, 45};
    H5::DataSpace space2(2, dims2);
    file.createDataSet("class_counts", H5::PredType::NATIVE_UINT8, space2)
        .write(all_counts.data(), H5::PredType::NATIVE_UINT8);

    file.createDataSet("outcome", H5::PredType::NATIVE_INT8, space1)
        .write(all_outcome.data(), H5::PredType::NATIVE_INT8);
  }

  std::cout << "Wrote " << n << " positions to " << output_file << std::endl;

  return 0;
}
