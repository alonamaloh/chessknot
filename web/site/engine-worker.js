/**
 * Web Worker for the ChessKnot engine
 */

let engine = null;
let board = null;
let isReady = false;

async function init(distBase) {
    distBase = distBase || './dist/';
    try {
        importScripts(`${distBase}engine.js`);
        engine = await ChessKnotEngine({
            locateFile: (path) => `${distBase}${path}`
        });

        const version = engine.getEngineVersion();
        console.log(`Engine version: ${version}`);

        board = engine.getInitialBoard();
        isReady = true;
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: `Failed to initialize engine: ${err.message}` });
    }
}

function loadNNModel(data) {
    if (!engine) return;
    engine.loadNNModel(data);
}

function getLegalMoves() {
    if (!engine || !board) return [];
    return engine.getLegalMoves(board);
}

function makeMove(from, to) {
    if (!engine || !board) return null;
    board = engine.makeMove(board, from, to);
    return getBoardState();
}

function resetBoard() {
    if (!engine) return;
    board = engine.getInitialBoard();
    return getBoardState();
}

function getBoardState() {
    if (!board) return null;
    return {
        whiteLo: board.getWhiteLo(),
        whiteHi: board.getWhiteHi(),
        blackLo: board.getBlackLo(),
        blackHi: board.getBlackHi(),
        whiteToMove: board.isWhiteToMove()
    };
}

function getLockedPieces() {
    if (!engine || !board) return { whiteLo: 0, whiteHi: 0, blackLo: 0, blackHi: 0 };
    return engine.getLockedPieces(board);
}

function search(softTime, hardTime) {
    if (!engine || !board) {
        return { error: 'Engine not ready' };
    }

    try {
        const result = engine.search(board, 100, softTime || 3, hardTime || 10);

        return {
            bestFrom: result.bestFrom,
            bestTo: result.bestTo,
            score: result.score,
            depth: result.depth,
            nodes: result.nodes
        };
    } catch (err) {
        console.error('Worker: search error:', err);
        return { error: err.message || 'Search failed' };
    }
}

self.onmessage = function(e) {
    const { id, type, data } = e.data;

    let response = { id, type };

    try {
        switch (type) {
            case 'init':
                init(data && data.distBase);
                return;

            case 'loadNNModel':
                loadNNModel(data.data);
                response.success = true;
                break;

            case 'getLegalMoves':
                response.moves = getLegalMoves();
                break;

            case 'makeMove':
                response.board = makeMove(data.from, data.to);
                break;

            case 'resetBoard':
                response.board = resetBoard();
                break;

            case 'getBoard':
                response.board = getBoardState();
                break;

            case 'getLockedPieces':
                response.locked = getLockedPieces();
                break;

            case 'search':
                response.result = search(data.softTime, data.hardTime);
                break;

            case 'stop':
                // Without pthreads, search runs synchronously in worker
                // and can't be interrupted. TimeControl handles the time limit.
                response.success = true;
                break;

            case 'getStatus':
                response.status = {
                    nnModel: engine ? engine.hasNNModel() : false
                };
                response.ready = isReady;
                break;

            default:
                response.error = `Unknown message type: ${type}`;
        }
    } catch (err) {
        response.error = err.message;
    }

    postMessage(response);
};
