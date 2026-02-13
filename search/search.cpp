#include "search.hpp"
#include "../core/policy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace search {

// Policy logits from training_data7.policy (225 = 15x15 move classes)
// moveClass = (NEIGHBOR_OFFSET[w_before]+b_before)*15 + (NEIGHBOR_OFFSET[w_after]+b_after)
static constexpr float POLICY_LOGITS[225] = {
  +6.171f, +5.932f, +3.982f, +1.823f, -3.342f, +5.791f, +5.505f, +3.172f, +1.274f, +5.237f, +3.571f, +0.935f, +3.930f, +1.518f, -0.123f,
  -3.342f, +8.814f, +6.383f, +3.999f, -0.346f, -3.342f, +8.151f, +5.160f, +2.601f, -3.342f, +5.993f, +2.024f, -3.342f, +2.493f, -3.342f,
  -3.342f, -3.342f, +8.978f, +5.550f, +2.180f, -3.342f, -3.342f, +7.132f, +2.798f, -3.342f, -3.342f, +3.615f, -3.342f, -3.342f, -3.342f,
  -3.342f, -3.342f, -3.342f, +6.447f, +2.433f, -3.342f, -3.342f, -3.342f, +3.715f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f,
  -3.342f, -3.342f, -3.342f, -3.342f, +2.359f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f,
  +5.715f, +5.929f, +4.219f, +2.015f, -2.648f, +6.709f, +5.814f, +3.509f, +1.028f, +5.789f, +3.400f, +1.293f, +3.617f, +0.629f, -0.164f,
  -3.342f, +8.269f, +5.791f, +3.302f, -1.396f, -3.342f, +7.984f, +4.998f, +2.219f, -3.342f, +5.672f, +2.293f, -3.342f, +2.585f, -3.342f,
  -3.342f, -3.342f, +7.200f, +3.892f, +0.242f, -3.342f, -3.342f, +6.261f, +2.223f, -3.342f, -3.342f, +2.968f, -3.342f, -3.342f, -3.342f,
  -3.342f, -3.342f, -3.342f, +3.787f, -0.164f, -3.342f, -3.342f, -3.342f, +2.493f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f, -3.342f,
  +4.783f, +4.326f, +2.593f, +0.372f, -3.342f, +5.187f, +4.211f, +1.957f, -0.777f, +5.701f, +2.079f, -0.508f, +2.950f, -0.123f, -1.396f,
  -3.342f, +6.354f, +3.816f, +1.412f, -1.955f, -3.342f, +6.058f, +3.376f, +0.935f, -3.342f, +4.285f, +1.191f, -3.342f, +1.243f, -3.342f,
  -3.342f, -3.342f, +3.581f, +0.443f, -3.342f, -3.342f, -3.342f, +2.779f, -0.251f, -3.342f, -3.342f, +0.487f, -3.342f, -3.342f, -3.342f,
  +3.719f, +3.181f, +1.254f, -0.009f, -3.342f, +2.785f, +2.635f, -0.083f, -2.648f, +2.779f, -0.046f, -3.342f, +1.065f, -1.396f, -1.955f,
  -3.342f, +3.567f, +1.350f, -1.262f, -3.342f, -3.342f, +3.328f, +0.976f, -1.732f, -3.342f, +1.642f, -2.243f, -3.342f, -0.857f, -3.342f,
  +0.817f, -1.396f, -2.648f, -3.342f, -3.342f, -0.123f, -1.550f, -2.648f, -3.342f, -1.262f, -2.648f, -3.342f, -2.243f, -3.342f, -3.342f,
};

// --- TimeControl ---

void TimeControl::start() {
  start_time_ = std::chrono::steady_clock::now();
  next_check = (hard_node_limit > 0)
    ? std::min(hard_node_limit, CHECK_INTERVAL)
    : (hard_time_seconds > 0 ? CHECK_INTERVAL : UINT64_MAX);
}

void TimeControl::check(std::uint64_t nodes) {
  if (hard_node_limit > 0 && nodes >= hard_node_limit)
    throw SearchInterrupted{};
  if (hard_time_seconds > 0 && elapsed_seconds() >= hard_time_seconds)
    throw SearchInterrupted{};
  next_check = nodes + CHECK_INTERVAL;
  if (hard_node_limit > 0 && next_check > hard_node_limit)
    next_check = hard_node_limit;
}

bool TimeControl::exceeded_soft(std::uint64_t nodes) const {
  if (soft_node_limit > 0 && nodes >= soft_node_limit) return true;
  if (soft_time_seconds > 0 && elapsed_seconds() >= soft_time_seconds) return true;
  return false;
}

double TimeControl::elapsed_seconds() const {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(now - start_time_).count();
}

TimeControl TimeControl::with_nodes(std::uint64_t soft, std::uint64_t hard) {
  TimeControl tc;
  tc.soft_node_limit = soft;
  tc.hard_node_limit = (hard > 0) ? hard : (soft > 0 ? soft * 5 : 0);
  return tc;
}

TimeControl TimeControl::with_time(double soft_sec, double hard_sec) {
  TimeControl tc;
  tc.soft_time_seconds = soft_sec;
  tc.hard_time_seconds = hard_sec;
  return tc;
}

// --- Random eval ---

int random_eval(const Board& board) {
  std::uint64_t h = board.hash() + 1;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return static_cast<int>(static_cast<std::int64_t>(h) % 10001);
}

// --- Searcher ---

Searcher::Searcher() : tt_(64), eval_(random_eval) {}

int Searcher::negamax(const Board& board, int depth, int alpha, int beta, int ply) {
  // Hard limits
  if (nodes_ >= tc_.next_check)
    tc_.check(nodes_);

  nodes_++;

  // Repetition detection: check search path
  std::uint64_t h = board.hash();
  if (ply < MAX_PLY)
    hash_stack_[ply] = h;

  // Check within search tree (step by 2 since same side must be on move)
  for (int back = 2; back <= ply; back += 2) {
    if (hash_stack_[ply - back] == h)
      return 0;
  }
  // Check game history (positions before search root, reverse chronological).
  // Our hash encodes the side to move (white/black bitboards are distinct),
  // so opposite-side positions won't match. Just scan all entries.
  for (auto gh : game_history_) {
    if (gh == h)
      return 0;
  }

  // TT probe
  std::uint64_t key = h;
  TTEntry tt_entry;
  CompactMove tt_compact = 0;

  if (tt_.probe(key, tt_entry)) {
    tt_compact = tt_entry.best_move;
    if (tt_entry.depth >= depth) {
      int tt_score = tt_entry.score;
      switch (tt_entry.flag) {
        case TTFlag::EXACT:
          return tt_score;
        case TTFlag::LOWER_BOUND:
          if (tt_score >= beta) return tt_score;
          alpha = std::max(alpha, tt_score);
          break;
        case TTFlag::UPPER_BOUND:
          if (tt_score <= alpha) return tt_score;
          beta = std::min(beta, tt_score);
          break;
        default:
          break;
      }
    }
  }

  // Generate moves
  MoveList moves;
  generateMoves(board, moves);

  // No moves = loss
  if (moves.empty())
    return mated_score(ply);

  // Leaf node
  if (depth <= 0)
    return eval_(board);

  // Move ordering: TT move first, then killers, then history
  auto insert_pos = moves.begin();

  if (tt_compact != 0) {
    for (auto it = insert_pos; it != moves.end(); ++it) {
      if (compactMatches(tt_compact, *it)) {
        std::swap(*insert_pos, *it);
        ++insert_pos;
        break;
      }
    }
  }

  if (ply < MAX_PLY) {
    for (int k = 0; k < 2; ++k) {
      Move killer = killers_[ply][k];
      if (killer.from == 0 && killer.to == 0) continue;
      for (auto it = insert_pos; it != moves.end(); ++it) {
        if (it->from == killer.from && it->to == killer.to) {
          std::swap(*insert_pos, *it);
          ++insert_pos;
          break;
        }
      }
    }
  }

  // Sort remaining by history (lower = better, tried first)
  std::sort(insert_pos, moves.end(), [this](const Move& a, const Move& b) {
    return history_[a.from][a.to] < history_[b.from][b.to];
  });

  int best_score = -SCORE_INFINITE;
  Move best_move;
  TTFlag flag = TTFlag::UPPER_BOUND;
  bool is_first = true;

  for (const Move& move : moves) {
    Board child = makeMove(board, move);
    int score;

    if (is_first) {
      score = -negamax(child, depth - 1, -beta, -alpha, ply + 1);
    } else {
      // LMR: reduce late moves based on depth and policy logit
      int reduction = 0;
      if (depth >= 3) {
        float logit = POLICY_LOGITS[moveClass(board, move)];
        if (logit >= 6.0f) reduction = 0;
        else if (logit >= 2.0f) reduction = 1;
        else reduction = 1 + (depth >= 5 ? 1 : 0);
      }
      // PVS: null-window search, possibly at reduced depth
      score = -negamax(child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
      if (score > alpha)
        score = -negamax(child, depth - 1, -beta, -alpha, ply + 1);
    }

    if (score > best_score) {
      best_score = score;
      best_move = move;

      if (score > alpha) {
        alpha = score;
        flag = TTFlag::EXACT;

        if (alpha >= beta) {
          flag = TTFlag::LOWER_BOUND;
          // Killer move
          if (ply < MAX_PLY) {
            if (killers_[ply][0].from != move.from || killers_[ply][0].to != move.to) {
              killers_[ply][1] = killers_[ply][0];
              killers_[ply][0] = move;
            }
          }
          // History: penalize first move, reward cutoff move
          if (!is_first) {
            Move first = moves[0];
            if (history_[first.from][first.to] < INT16_MAX)
              history_[first.from][first.to]++;
            if (history_[move.from][move.to] > INT16_MIN)
              history_[move.from][move.to]--;
          }
          break;
        }
      }
    }
    is_first = false;
  }

  // Store in TT
  tt_.store(key, best_score, depth, flag, best_move);
  return best_score;
}

void Searcher::extract_pv(const Board& board, std::vector<Move>& pv, int max_depth) {
  pv.clear();
  Board pos = board;
  std::uint64_t seen[64];
  int seen_count = 0;

  for (int i = 0; i < max_depth && seen_count < 64; ++i) {
    std::uint64_t key = pos.hash();
    for (int j = 0; j < seen_count; ++j)
      if (seen[j] == key) return;
    seen[seen_count++] = key;

    TTEntry entry;
    if (!tt_.probe(key, entry) || entry.best_move == 0) return;

    MoveList moves;
    generateMoves(pos, moves);
    Move* found = nullptr;
    for (Move& m : moves) {
      if (compactMatches(entry.best_move, m)) {
        found = &m;
        break;
      }
    }
    if (!found) return;

    pv.push_back(*found);
    pos = makeMove(pos, *found);
  }
}

SearchResult Searcher::search_root(const Board& board, MoveList& moves, int depth) {
  SearchResult result;
  result.depth = depth;

  hash_stack_[0] = board.hash();

  int alpha = -SCORE_INFINITE;
  int beta = SCORE_INFINITE;

  for (std::size_t i = 0; i < moves.size(); ++i) {
    Board child = makeMove(board, moves[i]);
    int score = -negamax(child, depth - 1, -beta, -alpha, 1);

    if (score > alpha) {
      alpha = score;
      result.best_move = moves[i];
      result.score = score;

      if (i > 0)
        std::rotate(moves.begin(), moves.begin() + i, moves.begin() + i + 1);
    }
  }

  result.nodes = nodes_;

  // Extract PV
  result.pv.push_back(result.best_move);
  Board child = makeMove(board, result.best_move);
  std::vector<Move> continuation;
  extract_pv(child, continuation, depth - 1);
  result.pv.insert(result.pv.end(), continuation.begin(), continuation.end());

  return result;
}

SearchResult Searcher::search(const Board& board, int max_depth, const TimeControl& tc) {
  nodes_ = 0;
  tt_.new_search();
  tc_ = tc;
  tc_.start();
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
  std::memset(hash_stack_, 0, sizeof(hash_stack_));

  SearchResult result;

  MoveList root_moves;
  generateMoves(board, root_moves);

  if (root_moves.empty()) {
    result.score = mated_score(0);
    return result;
  }

  if (root_moves.size() == 1) {
    result.best_move = root_moves[0];
    result.depth = 1;
    result.score = -eval_(makeMove(board, root_moves[0]));
    return result;
  }

  for (int depth = 1; depth <= max_depth; ++depth) {
    try {
      result = search_root(board, root_moves, depth);
      result.depth = depth;
    } catch (const SearchInterrupted&) {
      break;
    }

    if (verbose_) {
      double elapsed = tc_.elapsed_seconds();
      std::uint64_t nps = elapsed > 0.001 ? static_cast<std::uint64_t>(nodes_ / elapsed) : 0;
      std::cout << "depth " << depth
                << " score " << result.score
                << " nodes " << nodes_
                << " nps " << nps
                << " pv";
      for (const Move& m : result.pv)
        std::cout << " " << m;
      std::cout << std::endl;
    }

    if (is_mate_score(result.score))
      break;
    if (tc_.exceeded_soft(nodes_))
      break;
  }

  return result;
}

} // namespace search
