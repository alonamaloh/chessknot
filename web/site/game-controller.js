/**
 * Game controller — manages game state, history, and coordinates UI with engine
 */

import { BoardUI } from './board-ui.js';

export class GameController {
    constructor(canvas, engine, statusElement = null) {
        this.boardUI = new BoardUI(canvas);
        this.engine = engine;
        this.statusElement = statusElement;

        // Game state
        this.history = [];        // Array of { board, from, to }
        this.redoStack = [];
        this.legalMoves = [];     // Current legal moves [{from, to}, ...]

        // Settings
        this.humanColor = 'white';  // 'white', 'black', or 'both'
        this.secondsPerMove = 3.0;
        this.autoPlay = true;

        // State flags
        this.isThinking = false;
        this.gameOver = false;
        this.winner = null;
        this._aborting = false;

        // Callbacks
        this.onMove = null;
        this.onGameOver = null;
        this.onThinkingStart = null;
        this.onThinkingEnd = null;
        this.onStatusUpdate = null;
        this.onSearchInfo = null;
        this.onModeChange = null;

        // Set up board click handler
        this.boardUI.onClick = (square) => this._handleSquareClick(square);
    }

    async init() {
        if (!this.engine.isReady) {
            await this.engine.init();
        }
        await this.newGame();
    }

    async abortSearch() {
        if (!this.isThinking) return;
        this._aborting = true;
        this.engine.stopSearch();
        while (this.isThinking) {
            await this._sleep(50);
        }
        this._aborting = false;
    }

    async newGame() {
        await this.abortSearch();

        if (this.humanColor === 'black') {
            this.humanColor = 'white';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        } else if (this.humanColor === 'none') {
            // Keep auto mode
        }

        this.history = [];
        this.redoStack = [];
        this.gameOver = false;
        this.winner = null;
        this.boardUI.setSelected(null);
        this.boardUI.setLegalTargets([]);
        this.boardUI.clearLastMove();

        const board = await this.engine.resetBoard();
        this._updateFromBoard(board);
        await this._updateLegalMoves();
        this._updateStatus('New game');
    }

    _updateFromBoard(board) {
        this.boardUI.setPosition(
            board.whiteLo, board.whiteHi,
            board.blackLo, board.blackHi,
            board.whiteToMove
        );
    }

    async _updateLegalMoves() {
        this.legalMoves = await this.engine.getLegalMoves();

        if (this.legalMoves.length === 0) {
            const board = await this.engine.getBoard();
            const loser = board.whiteToMove ? 'white' : 'black';
            const winner = loser === 'white' ? 'black' : 'white';
            this._setGameOver(winner, 'no moves');
        }

        await this._updateLockedPieces();
    }

    async _updateLockedPieces() {
        const locked = await this.engine.getLockedPieces();
        this.boardUI.setLockedPieces(
            locked.whiteLo, locked.whiteHi,
            locked.blackLo, locked.blackHi
        );
    }

    _setGameOver(winner, reason) {
        this.gameOver = true;
        this.winner = winner;

        if (winner === 'draw') {
            this._updateStatus('Draw!');
        } else {
            this._updateStatus(`${winner === 'white' ? 'White' : 'Black'} wins!`);
        }

        if (this.onGameOver) {
            this.onGameOver(winner, reason);
        }
    }

    async _handleSquareClick(square) {
        if (this.gameOver || this.isThinking) return;

        const board = await this.engine.getBoard();
        if (!this._isHumanTurn(board.whiteToMove)) return;

        // If clicking on own piece, select it
        if (this.boardUI.hasPieceToMove(square)) {
            this.boardUI.setSelected(square);
            // Show legal targets from this square
            const targets = this.legalMoves
                .filter(m => m.from === square)
                .map(m => m.to);
            this.boardUI.setLegalTargets(targets);
            return;
        }

        // If a piece is selected and clicking on a legal target, make the move
        if (this.boardUI.selectedSquare !== null) {
            const move = this.legalMoves.find(
                m => m.from === this.boardUI.selectedSquare && m.to === square
            );
            if (move) {
                this.boardUI.setSelected(null);
                this.boardUI.setLegalTargets([]);
                await this._makeMove(move.from, move.to);
                return;
            }
        }

        // Clicked on empty/opponent — deselect
        this.boardUI.setSelected(null);
        this.boardUI.setLegalTargets([]);
    }

    _isHumanTurn(whiteToMove) {
        if (this.humanColor === 'none') return false;
        if (this.humanColor === 'both') return true;
        if (this.humanColor === 'white') return whiteToMove;
        return !whiteToMove;
    }

    async _makeMove(from, to, triggerAutoPlay = true) {
        this.redoStack = [];

        const prevBoard = await this.engine.getBoard();
        this.history.push({ board: { ...prevBoard }, from, to });

        const newBoard = await this.engine.makeMove(from, to);
        this._updateFromBoard(newBoard);
        this.boardUI.setSelected(null);
        this.boardUI.setLegalTargets([]);
        this.boardUI.setLastMove(from, to);

        if (this.onMove) {
            this.onMove({ from, to }, newBoard);
        }

        await this._updateLegalMoves();

        if (!this.gameOver) {
            const side = newBoard.whiteToMove ? 'White' : 'Black';
            this._updateStatus(`${side} to move`);
        }

        if (triggerAutoPlay && !this.gameOver && this.autoPlay && !this._isHumanTurn(newBoard.whiteToMove)) {
            await this._engineMove();
        }
    }

    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async _engineMove() {
        if (this.gameOver || this.isThinking) return;

        this.isThinking = true;
        if (this.onThinkingStart) this.onThinkingStart();
        this._updateStatus('Engine thinking...');

        try {
            const softTime = this.secondsPerMove / 3;
            const hardTime = this.secondsPerMove * 2;

            const result = await this.engine.search(softTime, hardTime);

            if (this._aborting) return;

            if (result.error) {
                console.error('Engine error:', result.error);
                this._updateStatus('Engine error: ' + result.error);
                return;
            }

            // Report search info
            if (this.onSearchInfo) {
                this.onSearchInfo({
                    depth: result.depth,
                    score: result.score,
                    nodes: result.nodes
                });
            }

            // Ensure minimum 200ms so user can see what happened
            await this._sleep(200);

            if (result.bestFrom !== undefined && result.bestTo !== undefined) {
                this.isThinking = false;
                if (this.onThinkingEnd) this.onThinkingEnd();
                await this._makeMove(result.bestFrom, result.bestTo);
                return;
            } else {
                this._updateStatus('No move found');
            }
        } catch (err) {
            console.error('Search exception:', err);
            this._updateStatus('Search error: ' + err.message);
        } finally {
            this.isThinking = false;
            if (this.onThinkingEnd) this.onThinkingEnd();
        }
    }

    async undo() {
        if (this.history.length === 0) return;
        await this.abortSearch();

        const last = this.history.pop();
        this.redoStack.push(last);

        // Restore by replaying from initial position
        await this._replayHistory();

        this.gameOver = false;
        this.winner = null;

        if (this.humanColor !== 'both' && this.humanColor !== 'none') {
            const board = await this.engine.getBoard();
            this.humanColor = board.whiteToMove ? 'white' : 'black';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this._updateStatus('Move undone');
    }

    async redo() {
        if (this.redoStack.length === 0 || this.isThinking) return;

        const entry = this.redoStack.pop();
        this.history.push(entry);

        const newBoard = await this.engine.makeMove(entry.from, entry.to);
        this._updateFromBoard(newBoard);
        this.boardUI.setSelected(null);
        this.boardUI.setLegalTargets([]);

        await this._updateLegalMoves();
        this.boardUI.setLastMove(entry.from, entry.to);

        if (this.humanColor !== 'both') {
            this.humanColor = newBoard.whiteToMove ? 'white' : 'black';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this._updateStatus('Move redone');
    }

    async _replayHistory() {
        await this.engine.resetBoard();
        for (const entry of this.history) {
            await this.engine.makeMove(entry.from, entry.to);
        }
        const board = await this.engine.getBoard();
        this._updateFromBoard(board);
        this.boardUI.setSelected(null);
        this.boardUI.setLegalTargets([]);

        await this._updateLegalMoves();

        if (this.history.length > 0) {
            const last = this.history[this.history.length - 1];
            this.boardUI.setLastMove(last.from, last.to);
        }
    }

    async setHumanColor(color) {
        this.humanColor = color;

        if (!this.gameOver && !this.isThinking && this.autoPlay) {
            const board = await this.engine.getBoard();
            if (!this._isHumanTurn(board.whiteToMove)) {
                await this._engineMove();
            }
        }
    }

    flipBoard() {
        this.boardUI.setFlipped(!this.boardUI.flipped);
    }

    stopSearch() {
        if (this.engine) {
            this.engine.stopSearch();
        }
    }

    setSecondsPerMove(seconds) {
        this.secondsPerMove = seconds;
    }

    getGameNotation() {
        let notation = '';
        for (let i = 0; i < this.history.length; i++) {
            const moveNum = Math.floor(i / 2) + 1;
            if (i % 2 === 0) notation += `${moveNum}. `;
            const { from, to } = this.history[i];
            const fromStr = String.fromCharCode(97 + (from % 8)) + (Math.floor(from / 8) + 1);
            const toStr = String.fromCharCode(97 + (to % 8)) + (Math.floor(to / 8) + 1);
            notation += `${fromStr}-${toStr} `;
        }
        return notation.trim();
    }

    getMoveHistoryForDisplay() {
        const historyMoves = this.history.map(h => {
            const fromStr = String.fromCharCode(97 + (h.from % 8)) + (Math.floor(h.from / 8) + 1);
            const toStr = String.fromCharCode(97 + (h.to % 8)) + (Math.floor(h.to / 8) + 1);
            return `${fromStr}-${toStr}`;
        });
        const redoMoves = this.redoStack.slice().reverse().map(h => {
            const fromStr = String.fromCharCode(97 + (h.from % 8)) + (Math.floor(h.from / 8) + 1);
            const toStr = String.fromCharCode(97 + (h.to % 8)) + (Math.floor(h.to / 8) + 1);
            return `${fromStr}-${toStr}`;
        });
        return { history: historyMoves, redo: redoMoves };
    }

    getUndoRedoState() {
        return {
            canUndo: this.history.length > 0 && !this.isThinking,
            canRedo: this.redoStack.length > 0 && !this.isThinking
        };
    }

    _updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
        if (this.onStatusUpdate) {
            this.onStatusUpdate(message);
        }
    }

    resize(size) {
        this.boardUI.resize(size);
    }

    async getBoard() {
        return await this.engine.getBoard();
    }
}
