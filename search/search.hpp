#pragma once

#include "../core/board.hpp"
#include "../core/movegen.hpp"
#include "tt.hpp"
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace search {

constexpr int MAX_PLY = 128;
constexpr int SCORE_INFINITE = 32000;
constexpr int SCORE_MATE = 30000;

inline int mate_score(int ply) { return SCORE_MATE - ply; }
inline int mated_score(int ply) { return -SCORE_MATE + ply; }
inline bool is_mate_score(int score) { return score > 29000 || score < -29000; }

// Evaluation function: takes a board (white to move), returns score for side to move
using EvalFunc = std::function<int(const Board&)>;

// Random eval: reproducible pseudo-random score in [-10000, +10000]
int random_eval(const Board& board);

struct SearchResult {
  Move best_move;
  int score = 0;
  int depth = 0;
  std::uint64_t nodes = 0;
  std::vector<Move> pv;
};

struct SearchInterrupted : std::exception {
  const char* what() const noexcept override { return "search interrupted"; }
};

struct TimeControl {
  std::uint64_t soft_node_limit = 0;
  std::uint64_t hard_node_limit = 0;
  double soft_time_seconds = 0;
  double hard_time_seconds = 0;

  std::uint64_t next_check = 0;

  void start();
  void check(std::uint64_t nodes);
  bool exceeded_soft(std::uint64_t nodes) const;
  double elapsed_seconds() const;

  static TimeControl with_nodes(std::uint64_t soft, std::uint64_t hard = 0);
  static TimeControl with_time(double soft_sec, double hard_sec = 0);

private:
  std::chrono::steady_clock::time_point start_time_;
  static constexpr std::uint64_t CHECK_INTERVAL = 4096;
};

class Searcher {
public:
  Searcher();

  void set_eval(EvalFunc eval) { eval_ = std::move(eval); }
  void set_tt_size(std::size_t mb) { tt_ = TranspositionTable(mb); }
  void clear_tt() { tt_.clear(); }
  void set_verbose(bool v) { verbose_ = v; }

  // Set game history for repetition detection before search starts.
  // Hashes should be from alternating sides (as stored: always white-to-move).
  void set_history(const std::vector<std::uint64_t>& hashes) { game_history_ = hashes; }

  SearchResult search(const Board& board, int max_depth = 100,
                      const TimeControl& tc = TimeControl{});

  std::uint64_t last_nodes() const { return nodes_; }

private:
  SearchResult search_root(const Board& board, MoveList& moves, int depth);
  int negamax(const Board& board, int depth, int alpha, int beta, int ply);
  void extract_pv(const Board& board, std::vector<Move>& pv, int max_depth);

  TranspositionTable tt_;
  EvalFunc eval_;
  std::uint64_t nodes_ = 0;
  TimeControl tc_;
  bool verbose_ = false;

  Move killers_[MAX_PLY][2] = {};
  std::int16_t history_[64][64] = {};
  std::uint64_t hash_stack_[MAX_PLY] = {};

  // Game history (positions before search root) for repetition detection
  std::vector<std::uint64_t> game_history_;
};

} // namespace search
