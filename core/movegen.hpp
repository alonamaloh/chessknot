#pragma once

#include "board.hpp"
#include <cstdint>

// Generate all legal moves for the side to move (white).
std::size_t generateMoves(const Board& board, MoveList& moves);

// Perft: count leaf nodes at given depth
std::uint64_t perft(const Board& board, int depth);
