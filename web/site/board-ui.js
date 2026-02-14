/**
 * Canvas-based 8x8 board renderer for ChessKnot
 *
 * Square mapping: square = rank*8 + file, where rank 0 = row 1 (bottom), file 0 = column A
 * So square 0 = a1 (bottom-left), square 63 = h8 (top-right)
 *
 * Bitboards are 64-bit, stored as {lo, hi} pairs of uint32.
 * Bit i of the bitboard corresponds to square i.
 */

// Check if bit 'sq' is set in the {lo, hi} pair
function hasBit(lo, hi, sq) {
    if (sq < 32) return (lo & (1 << sq)) !== 0;
    return (hi & (1 << (sq - 32))) !== 0;
}

// Preload piece SVG images (Cburnett set, CC BY-SA 3.0)
const PIECE_IMAGES = {};
const PIECE_TYPES = ['K', 'Q', 'R', 'B', 'N', 'P'];
const PIECE_COLORS = ['w', 'b'];
let _piecesLoaded = 0;
let _onPiecesLoaded = null;

const _piecesBase = new URL('./pieces/', import.meta.url).href;

for (const color of PIECE_COLORS) {
    PIECE_IMAGES[color] = {};
    for (const type of PIECE_TYPES) {
        const img = new Image();
        img.src = `${_piecesBase}${color}${type}.svg`;
        img.onload = () => { if (++_piecesLoaded === 12 && _onPiecesLoaded) _onPiecesLoaded(); };
        PIECE_IMAGES[color][type] = img;
    }
}

export class BoardUI {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.size = Math.min(canvas.width, canvas.height);
        this.squareSize = this.size / 8;

        // Board state (bitboards from engine, game perspective)
        this.whiteLo = 0;
        this.whiteHi = 0;
        this.blackLo = 0;
        this.blackHi = 0;
        this.whiteToMove = true;

        // UI state
        this.selectedSquare = null;    // Currently selected piece (0-63)
        this.legalTargets = [];        // Squares this piece can move to
        this.lastMove = null;          // { from, to }
        this.flipped = false;
        this.pieceMap = null;       // Array(64) of piece type letters, or null

        // Re-render once all piece images are loaded
        if (_piecesLoaded < 12) _onPiecesLoaded = () => this.render();

        // Locked pieces (no legal moves) — bitboards in game perspective
        this.lockedWhiteLo = 0;
        this.lockedWhiteHi = 0;
        this.lockedBlackLo = 0;
        this.lockedBlackHi = 0;

        // Colors
        this.colors = {
            lightSquare: '#F0D9B5',
            darkSquare: '#B58863',
            whitePiece: '#FFFFFF',
            whitePieceStroke: '#333333',
            blackPiece: '#333333',
            blackPieceStroke: '#000000',
            selected: 'rgba(255, 255, 0, 0.5)',
            legalDot: 'rgba(0, 0, 0, 0.2)',
            lastMove: 'rgba(155, 199, 0, 0.4)',
        };

        // Event handling
        this.onClick = null;  // (square) => void

        this.canvas.addEventListener('click', (e) => this._handleClick(e));
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (e.changedTouches.length > 0) {
                this._handleTouch(e.changedTouches[0]);
            }
        });

        this.render();
    }

    setPosition(whiteLo, whiteHi, blackLo, blackHi, whiteToMove) {
        this.whiteLo = whiteLo >>> 0;
        this.whiteHi = whiteHi >>> 0;
        this.blackLo = blackLo >>> 0;
        this.blackHi = blackHi >>> 0;
        this.whiteToMove = whiteToMove;
        this.render();
    }

    setSelected(square) {
        this.selectedSquare = square;
        this.render();
    }

    setLegalTargets(targets) {
        this.legalTargets = targets || [];
        this.render();
    }

    setLastMove(from, to) {
        this.lastMove = { from, to };
        this.render();
    }

    clearLastMove() {
        this.lastMove = null;
        this.render();
    }

    setLockedPieces(wLo, wHi, bLo, bHi) {
        this.lockedWhiteLo = wLo >>> 0;
        this.lockedWhiteHi = wHi >>> 0;
        this.lockedBlackLo = bLo >>> 0;
        this.lockedBlackHi = bHi >>> 0;
        this.render();
    }

    setPieceMap(map) {
        this.pieceMap = map || null;
        this.render();
    }

    setFlipped(flipped) {
        this.flipped = flipped;
        this.render();
    }

    resize(size) {
        this.canvas.width = size;
        this.canvas.height = size;
        this.size = size;
        this.squareSize = size / 8;
        this.render();
    }

    render() {
        const ctx = this.ctx;
        const sq = this.squareSize;

        ctx.clearRect(0, 0, this.size, this.size);

        // Draw board squares
        for (let rank = 0; rank < 8; rank++) {
            for (let file = 0; file < 8; file++) {
                // Canvas position: rank 7 at top, rank 0 at bottom
                const displayRank = this.flipped ? rank : 7 - rank;
                const displayFile = this.flipped ? 7 - file : file;
                const x = displayFile * sq;
                const y = displayRank * sq;

                // (rank + file) even = light, odd = dark (with rank 0/file 0 = a1 = light)
                const isDark = (rank + file) % 2 === 1;
                ctx.fillStyle = isDark ? this.colors.darkSquare : this.colors.lightSquare;
                ctx.fillRect(x, y, sq, sq);
            }
        }

        // Highlight last move
        if (this.lastMove) {
            this._highlightSquare(this.lastMove.from, this.colors.lastMove);
            this._highlightSquare(this.lastMove.to, this.colors.lastMove);
        }

        // Highlight selected square
        if (this.selectedSquare !== null) {
            this._highlightSquare(this.selectedSquare, this.colors.selected);
        }

        // Draw pieces
        for (let square = 0; square < 64; square++) {
            const isWhite = hasBit(this.whiteLo, this.whiteHi, square);
            const isBlack = hasBit(this.blackLo, this.blackHi, square);

            if (isWhite || isBlack) {
                this._drawPiece(square, isWhite);
            }
        }

        // Draw legal move dots
        for (const target of this.legalTargets) {
            this._drawDot(target);
        }

        // Draw file/rank labels
        this._drawLabels();
    }

    _highlightSquare(square, color) {
        if (square === null || square < 0 || square > 63) return;
        const rank = Math.floor(square / 8);
        const file = square % 8;
        const displayRank = this.flipped ? rank : 7 - rank;
        const displayFile = this.flipped ? 7 - file : file;
        const x = displayFile * this.squareSize;
        const y = displayRank * this.squareSize;

        this.ctx.fillStyle = color;
        this.ctx.fillRect(x, y, this.squareSize, this.squareSize);
    }

    _drawPiece(square, isWhite) {
        const rank = Math.floor(square / 8);
        const file = square % 8;
        const displayRank = this.flipped ? rank : 7 - rank;
        const displayFile = this.flipped ? 7 - file : file;
        const x = displayFile * this.squareSize;
        const y = displayRank * this.squareSize;

        const pieceType = this.pieceMap && this.pieceMap[square];
        const color = isWhite ? 'w' : 'b';
        const locked = isWhite
            ? hasBit(this.lockedWhiteLo, this.lockedWhiteHi, square)
            : hasBit(this.lockedBlackLo, this.lockedBlackHi, square);

        if (locked) {
            this.ctx.save();
            this.ctx.shadowColor = 'rgba(220, 20, 20, 1)';
            this.ctx.shadowBlur = this.squareSize * 0.35;
        }

        if (pieceType && PIECE_IMAGES[color][pieceType] && PIECE_IMAGES[color][pieceType].complete) {
            const padding = this.squareSize * 0.05;
            const size = this.squareSize - padding * 2;
            // Draw twice when locked to intensify the glow
            if (locked) this.ctx.drawImage(PIECE_IMAGES[color][pieceType], x + padding, y + padding, size, size);
            this.ctx.drawImage(PIECE_IMAGES[color][pieceType], x + padding, y + padding, size, size);
        } else {
            // Fallback: colored circle
            const cx = x + this.squareSize / 2;
            const cy = y + this.squareSize / 2;
            const radius = this.squareSize * 0.4;
            const ctx = this.ctx;

            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);

            if (isWhite) {
                const grad = ctx.createRadialGradient(cx - radius * 0.3, cy - radius * 0.3, radius * 0.1, cx, cy, radius);
                grad.addColorStop(0, '#fff');
                grad.addColorStop(1, '#ccc');
                ctx.fillStyle = grad;
            } else {
                const grad = ctx.createRadialGradient(cx - radius * 0.3, cy - radius * 0.3, radius * 0.1, cx, cy, radius);
                grad.addColorStop(0, '#555');
                grad.addColorStop(1, '#222');
                ctx.fillStyle = grad;
            }
            ctx.fill();

            ctx.strokeStyle = isWhite ? this.colors.whitePieceStroke : this.colors.blackPieceStroke;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        if (locked) this.ctx.restore();
    }

    _drawDot(square) {
        const rank = Math.floor(square / 8);
        const file = square % 8;
        const displayRank = this.flipped ? rank : 7 - rank;
        const displayFile = this.flipped ? 7 - file : file;
        const x = displayFile * this.squareSize + this.squareSize / 2;
        const y = displayRank * this.squareSize + this.squareSize / 2;

        // Check if square is occupied (draw ring instead of dot)
        const occupied = hasBit(this.whiteLo, this.whiteHi, square) ||
                         hasBit(this.blackLo, this.blackHi, square);

        const ctx = this.ctx;
        if (occupied) {
            // Ring around occupied square
            ctx.beginPath();
            ctx.arc(x, y, this.squareSize * 0.4, 0, Math.PI * 2);
            ctx.strokeStyle = this.colors.legalDot;
            ctx.lineWidth = 3;
            ctx.stroke();
        } else {
            // Dot on empty square
            ctx.beginPath();
            ctx.arc(x, y, this.squareSize * 0.15, 0, Math.PI * 2);
            ctx.fillStyle = this.colors.legalDot;
            ctx.fill();
        }
    }

    _drawLabels() {
        const ctx = this.ctx;
        const sq = this.squareSize;
        ctx.font = `${sq * 0.15}px sans-serif`;
        ctx.textBaseline = 'bottom';
        ctx.textAlign = 'left';

        // File labels (a-h) along the bottom
        for (let file = 0; file < 8; file++) {
            const displayFile = this.flipped ? 7 - file : file;
            const x = displayFile * sq + sq * 0.05;
            const y = this.size - 1;
            const isDark = (0 + file) % 2 === 1; // rank 0
            const bottomRank = this.flipped ? 7 : 0;
            const isDarkActual = (bottomRank + file) % 2 === 1;
            ctx.fillStyle = isDarkActual ? this.colors.lightSquare : this.colors.darkSquare;
            ctx.fillText(String.fromCharCode(97 + file), x, y);
        }

        // Rank labels (1-8) along the left
        ctx.textBaseline = 'top';
        for (let rank = 0; rank < 8; rank++) {
            const displayRank = this.flipped ? rank : 7 - rank;
            const x = 2;
            const y = displayRank * sq + 1;
            const leftFile = this.flipped ? 7 : 0;
            const isDarkActual = (rank + leftFile) % 2 === 1;
            ctx.fillStyle = isDarkActual ? this.colors.lightSquare : this.colors.darkSquare;
            ctx.fillText(String(rank + 1), x, y);
        }
    }

    _handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        this._handleXY(x, y);
    }

    _handleTouch(touch) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const x = (touch.clientX - rect.left) * scaleX;
        const y = (touch.clientY - rect.top) * scaleY;
        this._handleXY(x, y);
    }

    _handleXY(x, y) {
        let displayFile = Math.floor(x / this.squareSize);
        let displayRank = Math.floor(y / this.squareSize);

        // Convert display coords to board coords
        const file = this.flipped ? 7 - displayFile : displayFile;
        const rank = this.flipped ? displayRank : 7 - displayRank;

        if (file < 0 || file > 7 || rank < 0 || rank > 7) return;

        const square = rank * 8 + file;
        if (this.onClick) {
            this.onClick(square);
        }
    }

    hasPieceToMove(square) {
        if (square < 0 || square > 63) return false;
        if (this.whiteToMove) {
            return hasBit(this.whiteLo, this.whiteHi, square);
        } else {
            return hasBit(this.blackLo, this.blackHi, square);
        }
    }
}
