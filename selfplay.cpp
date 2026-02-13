#include "core/board.hpp"
#include "core/features.hpp"
#include "core/movegen.hpp"
#include "nn/mlp.hpp"
#include "search/search.hpp"
#include <H5Cpp.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Data for one completed game
struct GameData {
  std::vector<std::uint64_t> white, black;
  std::vector<std::uint8_t> counts;  // flattened, 45 per position
  std::vector<std::int8_t> outcome;
};

// Random eval seeded per game
static search::EvalFunc makeRandomEval(std::uint64_t seed) {
  return [seed](const Board& b) -> int {
    std::uint64_t h = b.hash() ^ seed;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int>(static_cast<std::int64_t>(h) % 10001);
  };
}

// Play one self-play game. Returns: +1 first mover wins, -1 second mover wins, 0 draw.
int playGame(search::Searcher& searcher, const search::EvalFunc& eval,
             GameData& gd, int node_limit, int max_plies) {
  Board board;

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
  int flush_interval = 100;
  const char* output_file = "training_data.h5";
  const char* model_path = nullptr;

  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::cout << "Usage: selfplay [num_games] [nodes_per_move] [output_file] [threads] [flush_interval] [-m model.bin]\n"
              << "  num_games      Number of self-play games (default: 1000)\n"
              << "  nodes_per_move Node limit per search (default: 10000)\n"
              << "  output_file    HDF5 output path (default: training_data.h5)\n"
              << "  threads        Number of threads (default: hardware concurrency)\n"
              << "  flush_interval Flush to disk every N games (default: 100)\n"
              << "  -m model.bin   Use NN model for eval (default: random eval)\n";
    return 0;
  }

  // Parse positional args, then check for -m flag
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-m" && i + 1 < argc) {
      model_path = argv[++i];
    } else {
      positional.push_back(argv[i]);
    }
  }
  if (positional.size() > 0) num_games = std::atoi(positional[0].c_str());
  if (positional.size() > 1) node_limit = std::atoi(positional[1].c_str());
  if (positional.size() > 2) output_file = positional[2].c_str();
  if (positional.size() > 3) num_threads = std::atoi(positional[3].c_str());
  if (positional.size() > 4) flush_interval = std::atoi(positional[4].c_str());
  if (num_threads < 1) num_threads = 1;
  if (flush_interval < 1) flush_interval = 1;

  // Load model if specified (shared across threads, evaluate() is const)
  std::shared_ptr<nn::MLP> model;
  if (model_path) {
    model = std::make_shared<nn::MLP>(model_path);
    std::cout << "Using NN model: " << model_path << std::endl;
  }

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
      search::EvalFunc eval;
      if (model)
        eval = [&model](const Board& b) { return model->evaluate(b); };
      else
        eval = makeRandomEval(seeds[game]);
      int result = playGame(searcher, eval, gd, node_limit, 500);

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
