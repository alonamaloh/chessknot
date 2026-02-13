#include "core/board.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"
#include <H5Cpp.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Classify all 64 squares into 45 classes:
//   3 contents (empty=0, white=1, black=2) x
//   15 neighbor configs (w_neighbors, b_neighbors) with w+b <= 4
void computeClassCounts(const Board& board, std::uint8_t counts[45]) {
  Bb w = board.white;
  Bb aw = w << 8, bw = w >> 8;
  Bb cw = (w & ~FILE_H) << 1, dw = (w & ~FILE_A) >> 1;
  Bb ab_xor = aw ^ bw, ab_and = aw & bw;
  Bb Labc = ab_xor ^ cw, Mabc = ab_and | (ab_xor & cw);
  Bb Lw = Labc ^ dw, carry = Labc & dw;
  Bb Mw = Mabc ^ carry, Hw = Mabc & carry;
  Bb ew[5] = {~(Lw|Mw|Hw), Lw & ~Mw, Mw & ~Lw, Mw & Lw, Hw};

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

// Data for one completed game
struct GameData {
  std::vector<std::uint64_t> white, black;
  std::vector<std::uint8_t> counts;  // flattened, 45 per position
  std::vector<std::int8_t> outcome;
};

// Play one self-play game. Returns: +1 first mover wins, -1 second mover wins, 0 draw.
int playGame(search::Searcher& searcher, std::uint64_t seed,
             GameData& gd, int node_limit, int max_plies) {
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
  int game_result = 0;

  for (int ply = 0; ply < max_plies; ++ply) {
    std::uint64_t h = board.hash();

    int rep_count = 0;
    for (auto ph : hashes)
      if (ph == h) rep_count++;
    if (rep_count >= 2) break;
    hashes.push_back(h);

    MoveList moves;
    generateMoves(board, moves);
    if (moves.empty()) {
      game_result = (ply % 2 == 0) ? -1 : 1;
      break;
    }

    gd.white.push_back(board.white);
    gd.black.push_back(board.black);

    std::uint8_t cnts[45];
    computeClassCounts(board, cnts);
    gd.counts.insert(gd.counts.end(), cnts, cnts + 45);

    gd.outcome.push_back(0);

    std::vector<std::uint64_t> history;
    for (int i = static_cast<int>(hashes.size()) - 2; i >= 0; --i)
      history.push_back(hashes[i]);
    searcher.set_history(history);

    auto tc = search::TimeControl::with_nodes(node_limit);
    auto result = searcher.search(board, 100, tc);
    board = makeMove(board, result.best_move);
  }

  // Label all positions
  for (std::size_t i = 0; i < gd.outcome.size(); ++i) {
    if (game_result == 0)
      gd.outcome[i] = 0;
    else
      gd.outcome[i] = static_cast<std::int8_t>(
          (i % 2 == 0) ? game_result : -game_result);
  }

  return game_result;
}

// Manages incremental HDF5 writes using extensible datasets
class HDF5Writer {
public:
  explicit HDF5Writer(const char* path) : path_(path), written_(0) {
    H5::H5File file(path, H5F_ACC_TRUNC);

    hsize_t zero = 0, max_unlim = H5S_UNLIMITED;
    hsize_t chunk = 4096;

    H5::DSetCreatPropList plist1;
    plist1.setChunk(1, &chunk);

    H5::DataSpace space1(1, &zero, &max_unlim);
    file.createDataSet("white", H5::PredType::NATIVE_UINT64, space1, plist1);
    file.createDataSet("black", H5::PredType::NATIVE_UINT64, space1, plist1);
    file.createDataSet("outcome", H5::PredType::NATIVE_INT8, space1, plist1);

    hsize_t zero2[2] = {0, 45}, max2[2] = {H5S_UNLIMITED, 45};
    hsize_t chunk2[2] = {4096, 45};
    H5::DSetCreatPropList plist2;
    plist2.setChunk(2, chunk2);

    H5::DataSpace space2(2, zero2, max2);
    file.createDataSet("class_counts", H5::PredType::NATIVE_UINT8, space2, plist2);
  }

  // Flush a batch of completed games to disk
  void flush(std::vector<GameData>& games) {
    if (games.empty()) return;

    // Merge games into contiguous arrays
    std::size_t n = 0;
    for (auto& g : games) n += g.outcome.size();

    std::vector<std::uint64_t> w, b;
    std::vector<std::uint8_t> c;
    std::vector<std::int8_t> o;
    w.reserve(n); b.reserve(n); c.reserve(n * 45); o.reserve(n);

    for (auto& g : games) {
      w.insert(w.end(), g.white.begin(), g.white.end());
      b.insert(b.end(), g.black.begin(), g.black.end());
      c.insert(c.end(), g.counts.begin(), g.counts.end());
      o.insert(o.end(), g.outcome.begin(), g.outcome.end());
    }
    games.clear();

    // Append to HDF5
    H5::H5File file(path_, H5F_ACC_RDWR);
    hsize_t old_size = written_;
    hsize_t new_size = written_ + n;

    auto extend1 = [&](const char* name, const void* data, const H5::PredType& type) {
      auto ds = file.openDataSet(name);
      ds.extend(&new_size);

      H5::DataSpace fspace = ds.getSpace();
      hsize_t count = n;
      fspace.selectHyperslab(H5S_SELECT_SET, &count, &old_size);

      H5::DataSpace mspace(1, &count);
      ds.write(data, type, mspace, fspace);
    };

    extend1("white", w.data(), H5::PredType::NATIVE_UINT64);
    extend1("black", b.data(), H5::PredType::NATIVE_UINT64);
    extend1("outcome", o.data(), H5::PredType::NATIVE_INT8);

    // class_counts is 2D
    {
      auto ds = file.openDataSet("class_counts");
      hsize_t new_dims[2] = {new_size, 45};
      ds.extend(new_dims);

      H5::DataSpace fspace = ds.getSpace();
      hsize_t count[2] = {n, 45};
      hsize_t start[2] = {old_size, 0};
      fspace.selectHyperslab(H5S_SELECT_SET, count, start);

      H5::DataSpace mspace(2, count);
      ds.write(c.data(), H5::PredType::NATIVE_UINT8, mspace, fspace);
    }

    written_ = new_size;
  }

  std::size_t total_positions() const { return written_; }

private:
  const char* path_;
  std::size_t written_;
};

int main(int argc, char* argv[]) {
  int num_games = 1000;
  int node_limit = 10000;
  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  int flush_interval = 100;  // flush every N games
  const char* output_file = "training_data.h5";

  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::cout << "Usage: selfplay [num_games] [nodes_per_move] [output_file] [threads] [flush_interval]\n"
              << "  num_games      Number of self-play games (default: 1000)\n"
              << "  nodes_per_move Node limit per search (default: 10000)\n"
              << "  output_file    HDF5 output path (default: training_data.h5)\n"
              << "  threads        Number of threads (default: hardware concurrency)\n"
              << "  flush_interval Flush to disk every N games (default: 100)\n";
    return 0;
  }

  if (argc > 1) num_games = std::atoi(argv[1]);
  if (argc > 2) node_limit = std::atoi(argv[2]);
  if (argc > 3) output_file = argv[3];
  if (argc > 4) num_threads = std::atoi(argv[4]);
  if (argc > 5) flush_interval = std::atoi(argv[5]);
  if (num_threads < 1) num_threads = 1;
  if (flush_interval < 1) flush_interval = 1;

  std::cout << "Self-play: " << num_games << " games, "
            << node_limit << " nodes/move, "
            << num_threads << " threads, flush every "
            << flush_interval << " games, output: " << output_file << std::endl;

  std::mt19937_64 rng(42);
  std::vector<std::uint64_t> seeds(num_games);
  for (auto& s : seeds) s = rng();

  HDF5Writer writer(output_file);
  std::mutex writer_mutex;
  std::vector<GameData> pending_games;
  int pending_count = 0;

  std::atomic<int> next_game{0};
  std::atomic<int> total_wins{0}, total_losses{0}, total_draws{0};

  auto worker = [&](int /*tid*/) {
    search::Searcher searcher;
    searcher.set_tt_size(16);

    while (true) {
      int game = next_game.fetch_add(1);
      if (game >= num_games) break;

      GameData gd;
      int result = playGame(searcher, seeds[game], gd, node_limit, 500);

      if (result > 0) total_wins++;
      else if (result < 0) total_losses++;
      else total_draws++;

      // Submit completed game
      bool should_flush = false;
      {
        std::lock_guard<std::mutex> lock(writer_mutex);
        pending_games.push_back(std::move(gd));
        pending_count++;
        if (pending_count >= flush_interval) should_flush = true;
      }

      if (should_flush) {
        std::lock_guard<std::mutex> lock(writer_mutex);
        if (pending_count >= flush_interval) {
          writer.flush(pending_games);
          pending_count = 0;

          int done = total_wins + total_losses + total_draws;
          std::cout << "Game " << done << "/" << num_games
                    << "  W:" << total_wins.load() << " B:" << total_losses.load()
                    << " D:" << total_draws.load()
                    << "  positions:" << writer.total_positions() << std::endl;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t)
    threads.emplace_back(worker, t);
  for (auto& t : threads)
    t.join();

  // Flush remaining games
  {
    std::lock_guard<std::mutex> lock(writer_mutex);
    writer.flush(pending_games);
  }

  std::cout << "Done. " << writer.total_positions() << " positions from "
            << (total_wins + total_losses + total_draws) << " games"
            << "  W:" << total_wins.load() << " B:" << total_losses.load()
            << " D:" << total_draws.load() << std::endl;

  return 0;
}
