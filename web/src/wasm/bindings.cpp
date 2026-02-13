// Emscripten bindings for the ChessKnot engine

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <chrono>


#include "../../../core/board.hpp"
#include "../../../core/movegen.hpp"
#include "../../../search/search.hpp"
#include "../../../nn/mlp.hpp"

using namespace emscripten;

#define ENGINE_VERSION __DATE__ " " __TIME__

namespace {
    std::unique_ptr<nn::MLP> g_nn_model;
}

// Load neural network model from typed array
void loadNNModel(val typed_array) {
    unsigned int length = typed_array["length"].as<unsigned int>();

    std::vector<uint8_t> buffer(length);
    val memoryView = val(typed_memory_view(length, buffer.data()));
    memoryView.call<void>("set", typed_array);

    const char* path = "/tmp/nn_model.bin";
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(buffer.data(), 1, buffer.size(), f);
        fclose(f);
        try {
            g_nn_model = std::make_unique<nn::MLP>(path);
        } catch (...) {}
    }
}

bool hasNNModel() {
    return g_nn_model != nullptr;
}

// Board wrapper for JS — handles the "white always to move" convention.
// Internally, Board.white = side to move, Board.black = opponent.
// JSBoard tracks which color is actually to move in the game,
// and presents bitboards from the game perspective.
struct JSBoard {
    Board board;
    bool white_to_move;  // Game perspective

    JSBoard() : white_to_move(true) {}
    JSBoard(const Board& b, bool wtm) : board(b), white_to_move(wtm) {}

    // Bitboards from game perspective, split into lo/hi uint32 (embind doesn't support uint64)
    uint32_t getWhiteLo() const {
        uint64_t bb = white_to_move ? board.white : board.black;
        return static_cast<uint32_t>(bb);
    }
    uint32_t getWhiteHi() const {
        uint64_t bb = white_to_move ? board.white : board.black;
        return static_cast<uint32_t>(bb >> 32);
    }
    uint32_t getBlackLo() const {
        uint64_t bb = white_to_move ? board.black : board.white;
        return static_cast<uint32_t>(bb);
    }
    uint32_t getBlackHi() const {
        uint64_t bb = white_to_move ? board.black : board.white;
        return static_cast<uint32_t>(bb >> 32);
    }
    bool isWhiteToMove() const { return white_to_move; }
};

JSBoard getInitialBoard() {
    return JSBoard(Board(), true);
}

// Get legal moves — returns JS array of {from, to}
val getLegalMoves(const JSBoard& jsboard) {
    MoveList moves;
    generateMoves(jsboard.board, moves);

    val result = val::array();
    for (const auto& m : moves) {
        val move = val::object();
        move.set("from", static_cast<int>(m.from));
        move.set("to", static_cast<int>(m.to));
        result.call<void>("push", move);
    }
    return result;
}

// Make a move — returns new JSBoard with flipped side
JSBoard makeJSMove(const JSBoard& jsboard, int from, int to) {
    Board new_board = makeMove(jsboard.board, Move(from, to));
    return JSBoard(new_board, !jsboard.white_to_move);
}

// Search for best move
val doSearch(const JSBoard& jsboard, int max_depth, double soft_time, double hard_time) {
    val result = val::object();

    // Set up evaluation
    auto eval_func = [](const Board& b) -> int {
        if (g_nn_model) {
            return g_nn_model->evaluate(b);
        }
        return search::random_eval(b);
    };

    search::Searcher searcher;
    searcher.set_eval(eval_func);
    searcher.set_tt_size(16);

    // Stop flag is not usable without pthreads (single-threaded worker).
    // Search relies on TimeControl for termination.

    search::SearchResult sr;
    try {
        sr = searcher.search(jsboard.board, max_depth,
                             search::TimeControl::with_time(soft_time, hard_time));
    } catch (const search::SearchInterrupted&) {
        result.set("error", std::string("Search interrupted"));
        return result;
    } catch (const std::exception& e) {
        result.set("error", std::string("Search exception: ") + e.what());
        return result;
    }

    result.set("bestFrom", static_cast<int>(sr.best_move.from));
    result.set("bestTo", static_cast<int>(sr.best_move.to));
    result.set("score", sr.score);
    result.set("depth", sr.depth);
    result.set("nodes", static_cast<double>(sr.nodes));

    return result;
}

void stopSearch() {
    // No-op without pthreads. Search stops via TimeControl.
}

std::string getEngineVersion() {
    return ENGINE_VERSION;
}

EMSCRIPTEN_BINDINGS(chessknot_engine) {
    class_<JSBoard>("Board")
        .constructor<>()
        .function("getWhiteLo", &JSBoard::getWhiteLo)
        .function("getWhiteHi", &JSBoard::getWhiteHi)
        .function("getBlackLo", &JSBoard::getBlackLo)
        .function("getBlackHi", &JSBoard::getBlackHi)
        .function("isWhiteToMove", &JSBoard::isWhiteToMove);

    function("getInitialBoard", &getInitialBoard);
    function("getLegalMoves", &getLegalMoves);
    function("makeMove", &makeJSMove);
    function("search", &doSearch);
    function("loadNNModel", &loadNNModel);
    function("hasNNModel", &hasNNModel);
    function("stopSearch", &stopSearch);
    function("getEngineVersion", &getEngineVersion);
}
