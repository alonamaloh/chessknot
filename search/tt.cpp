#include "tt.hpp"
#include <bit>
#include <cstdlib>
#include <new>

TranspositionTable::TranspositionTable(std::size_t size_mb) {
  std::size_t bytes = size_mb * 1024 * 1024;
  std::size_t num_entries = bytes / sizeof(TTEntry);

  if (num_entries >= BUCKET_SIZE)
    num_entries = std::size_t{1} << (63 - std::countl_zero(num_entries));
  else
    num_entries = BUCKET_SIZE;

  entries_ = static_cast<TTEntry*>(std::aligned_alloc(64, num_entries * sizeof(TTEntry)));
  if (!entries_) throw std::bad_alloc();

  num_entries_ = num_entries;
  mask_ = num_entries - 1;
  clear();
}

TranspositionTable::~TranspositionTable() { std::free(entries_); }

TranspositionTable::TranspositionTable(TranspositionTable&& other) noexcept
    : entries_(other.entries_), num_entries_(other.num_entries_),
      mask_(other.mask_), current_generation_(other.current_generation_) {
  other.entries_ = nullptr;
  other.num_entries_ = 0;
  other.mask_ = 0;
}

TranspositionTable& TranspositionTable::operator=(TranspositionTable&& other) noexcept {
  if (this != &other) {
    std::free(entries_);
    entries_ = other.entries_;
    num_entries_ = other.num_entries_;
    mask_ = other.mask_;
    current_generation_ = other.current_generation_;
    other.entries_ = nullptr;
    other.num_entries_ = 0;
    other.mask_ = 0;
  }
  return *this;
}

void TranspositionTable::clear() {
  for (std::size_t i = 0; i < num_entries_; ++i)
    entries_[i] = TTEntry{};
  current_generation_ = 0;
}

bool TranspositionTable::probe(std::uint64_t key, TTEntry& entry) const {
  std::size_t index = key & mask_;
  std::uint32_t lock = static_cast<std::uint32_t>(key >> 32);

  for (int i = 0; i < BUCKET_SIZE; ++i) {
    const TTEntry& stored = entries_[index ^ i];
    if (stored.lock == lock && stored.flag != TTFlag::NONE) {
      entry = stored;
      return true;
    }
  }
  return false;
}

void TranspositionTable::store(std::uint64_t key, int score, int depth,
                                TTFlag flag, Move best_move) {
  std::size_t index = key & mask_;
  std::uint32_t lock = static_cast<std::uint32_t>(key >> 32);

  std::size_t best_slot = 0;
  int lowest_priority = 999999;

  for (int i = 0; i < BUCKET_SIZE; ++i) {
    const TTEntry& stored = entries_[index ^ i];
    if (stored.lock == lock) {
      best_slot = index ^ i;
      break;
    }
    int priority = stored.depth;
    if (stored.generation != current_generation_)
      priority -= 1000;
    if (priority < lowest_priority) {
      lowest_priority = priority;
      best_slot = index ^ i;
    }
  }

  TTEntry& entry = entries_[best_slot];
  entry.lock = lock;
  entry.score = static_cast<std::int16_t>(score);
  entry.depth = static_cast<std::int8_t>(depth);
  entry.flag = flag;
  entry.generation = current_generation_;
  entry.best_move = moveToCompact(best_move);
}
