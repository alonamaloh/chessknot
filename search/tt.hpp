#pragma once

#include "../core/board.hpp"
#include <cstdint>

enum class TTFlag : std::uint8_t {
  NONE = 0,
  EXACT = 1,
  LOWER_BOUND = 2,
  UPPER_BOUND = 3
};

using CompactMove = std::uint16_t;

inline CompactMove moveToCompact(Move m) {
  return static_cast<std::uint16_t>(m.from) | (static_cast<std::uint16_t>(m.to) << 8);
}

inline bool compactMatches(CompactMove c, Move m) {
  return c == moveToCompact(m);
}

struct TTEntry {
  std::uint32_t lock;
  CompactMove best_move;
  std::int16_t score;
  std::int8_t depth;
  TTFlag flag;
  std::uint8_t generation;
  std::uint8_t padding[5];

  TTEntry() : lock(0), best_move(0), score(0), depth(0),
              flag(TTFlag::NONE), generation(0), padding{} {}
};

static_assert(sizeof(TTEntry) == 16);

class TranspositionTable {
public:
  explicit TranspositionTable(std::size_t size_mb = 64);
  ~TranspositionTable();

  TranspositionTable(const TranspositionTable&) = delete;
  TranspositionTable& operator=(const TranspositionTable&) = delete;
  TranspositionTable(TranspositionTable&& other) noexcept;
  TranspositionTable& operator=(TranspositionTable&& other) noexcept;

  void clear();
  void new_search() { current_generation_++; }

  bool probe(std::uint64_t key, TTEntry& entry) const;
  void store(std::uint64_t key, int score, int depth, TTFlag flag, Move best_move);

  std::size_t size() const { return num_entries_; }

private:
  static constexpr int BUCKET_SIZE = 4;

  TTEntry* entries_ = nullptr;
  std::size_t num_entries_ = 0;
  std::size_t mask_ = 0;
  std::uint8_t current_generation_ = 0;
};
