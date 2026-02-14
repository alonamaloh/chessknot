/**
 * Main entry point for the ChessKnot web app
 */

const _v = new URL(import.meta.url).searchParams.get('v') || '';
const _q = _v ? `?v=${_v}` : '';

const { GameController } = await import(`./game-controller.js${_q}`);
const { getEngine } = await import(`./engine-api.js${_q}`);

let gameController = null;

async function init() {
    const canvas = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const loadingEl = document.getElementById('loading');
    const gameContainerEl = document.getElementById('game-container');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (gameContainerEl) gameContainerEl.style.display = 'none';

    try {
        updateLoadingStatus('Starting engine...');
        const distBase = new URL('./dist/', window.location.href).href;
        const engine = getEngine();
        await engine.init(distBase);

        // Load NN model
        updateLoadingStatus('Loading neural network...');
        try {
            const response = await fetch('./model_good.bin');
            if (response.ok) {
                const buffer = await response.arrayBuffer();
                await engine.loadNNModel(new Uint8Array(buffer));
                console.log('NN model loaded');
            }
        } catch (err) {
            console.warn('Could not load NN model:', err);
        }

        updateLoadingStatus('Starting game...');
        gameController = new GameController(canvas, engine, statusEl);
        await gameController.init();

        // Set up callbacks
        gameController.onMove = () => {
            updateMoveHistory();
            updateUndoRedoButtons();
        };

        gameController.onGameOver = (winner) => {
            showGameOver(winner);
        };

        gameController.onThinkingStart = () => {
            setThinkingIndicator(true);
            updateUndoRedoButtons();
        };

        gameController.onThinkingEnd = () => {
            setThinkingIndicator(false);
            updateUndoRedoButtons();
        };

        gameController.onSearchInfo = (info) => {
            updateSearchInfo(info);
        };

        gameController.onModeChange = () => {
            updateModeButtons();
        };

        setupEventHandlers();
        updateModeButtons();
        updateUndoRedoButtons();

        if (loadingEl) loadingEl.style.display = 'none';
        if (gameContainerEl) gameContainerEl.style.display = 'flex';

    } catch (err) {
        console.error('Initialization failed:', err);
        updateLoadingStatus(`Error: ${err.message}`);
    }
}

function updateLoadingStatus(message) {
    const el = document.getElementById('loading-status');
    if (el) el.textContent = message;
}

function updateModeButtons() {
    const btnEngineWhite = document.getElementById('btn-engine-white');
    const btnEngineBlack = document.getElementById('btn-engine-black');
    const btnTwoPlayer = document.getElementById('btn-two-player');
    const btnAuto = document.getElementById('btn-auto');

    if (!btnEngineWhite || !btnEngineBlack || !btnTwoPlayer || !btnAuto) return;

    btnEngineWhite.classList.remove('active');
    btnEngineBlack.classList.remove('active');
    btnTwoPlayer.classList.remove('active');
    btnAuto.classList.remove('active');

    switch (gameController.humanColor) {
        case 'black':
            btnEngineWhite.classList.add('active');
            break;
        case 'white':
            btnEngineBlack.classList.add('active');
            break;
        case 'both':
            btnTwoPlayer.classList.add('active');
            break;
        case 'none':
            btnAuto.classList.add('active');
            break;
    }
}

function updateUndoRedoButtons() {
    if (!gameController) return;
    const undoBtn = document.getElementById('btn-undo');
    const redoBtn = document.getElementById('btn-redo');
    const { canUndo, canRedo } = gameController.getUndoRedoState();
    if (undoBtn) undoBtn.disabled = !canUndo;
    if (redoBtn) redoBtn.disabled = !canRedo;
}

function setupEventHandlers() {
    // New game
    document.getElementById('btn-new-game').addEventListener('click', async () => {
        clearSearchInfo();
        await gameController.newGame();
        updateMoveHistory();
        updateUndoRedoButtons();
        updateModeButtons();
    });

    // Undo
    document.getElementById('btn-undo').addEventListener('click', async () => {
        await gameController.undo();
        updateMoveHistory();
        updateUndoRedoButtons();
    });

    // Redo
    document.getElementById('btn-redo').addEventListener('click', async () => {
        await gameController.redo();
        updateMoveHistory();
        updateUndoRedoButtons();
    });

    // Flip
    document.getElementById('btn-flip').addEventListener('click', () => {
        gameController.flipBoard();
    });

    // Stop
    document.getElementById('btn-stop').addEventListener('click', () => {
        gameController.stopSearch();
    });

    // Mode buttons
    document.getElementById('btn-engine-white').addEventListener('click', () => {
        gameController.setHumanColor('black');
        updateModeButtons();
    });
    document.getElementById('btn-engine-black').addEventListener('click', () => {
        gameController.setHumanColor('white');
        updateModeButtons();
    });
    document.getElementById('btn-two-player').addEventListener('click', () => {
        gameController.setHumanColor('both');
        updateModeButtons();
    });
    document.getElementById('btn-auto').addEventListener('click', () => {
        gameController.setHumanColor('none');
        updateModeButtons();
    });

    // Time per move
    const timeInput = document.getElementById('time-input');
    if (timeInput) {
        timeInput.addEventListener('change', () => {
            let value = parseFloat(timeInput.value);
            if (isNaN(value) || value < 0.1) value = 0.1;
            gameController.setSecondsPerMove(value);
        });
    }
}

function updateMoveHistory() {
    const historyEl = document.getElementById('move-history');
    if (!historyEl || !gameController) return;

    const { history, redo } = gameController.getMoveHistoryForDisplay();
    const allMoves = [...history, ...redo];

    if (allMoves.length === 0) {
        historyEl.innerHTML = '';
        return;
    }

    let html = '';
    for (let i = 0; i < allMoves.length; i++) {
        const moveNum = Math.floor(i / 2) + 1;
        const isRedo = i >= history.length;

        if (i % 2 === 0) html += `${moveNum}. `;

        if (isRedo) {
            html += `<span class="redo-move">${allMoves[i]}</span> `;
        } else {
            html += `${allMoves[i]} `;
        }
    }

    historyEl.innerHTML = html.trim();
    historyEl.scrollTop = historyEl.scrollHeight;
}

function showGameOver(winner) {
    let message;
    if (winner === 'draw') {
        message = 'Draw!';
    } else {
        message = `${winner === 'white' ? 'White' : 'Black'} wins!`;
    }
    setTimeout(() => alert(message), 100);
}

function setThinkingIndicator(thinking) {
    const stopBtn = document.getElementById('btn-stop');
    if (stopBtn) stopBtn.disabled = !thinking;
}

function clearSearchInfo() {
    const searchInfo = document.getElementById('search-info');
    if (searchInfo) searchInfo.style.display = 'none';
}

function updateSearchInfo(info) {
    const searchInfo = document.getElementById('search-info');
    if (!searchInfo) return;

    searchInfo.style.display = 'block';

    const summaryEl = document.getElementById('search-summary');
    if (summaryEl) {
        let scoreStr;
        const score = info.score;
        if (Math.abs(score) > 29000) {
            const mateIn = Math.ceil((30000 - Math.abs(score)) / 2);
            scoreStr = score > 0 ? `M${mateIn}` : `-M${mateIn}`;
        } else {
            scoreStr = (score / 100).toFixed(2);
            if (score >= 0) scoreStr = '+' + scoreStr;
        }

        const nodes = info.nodes || 0;
        const nodesStr = nodes >= 1000 ? (nodes / 1000).toFixed(0) + 'k' : nodes.toString();

        summaryEl.textContent = `depth ${info.depth} score ${scoreStr} nodes ${nodesStr}`;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

window.gameController = () => gameController;
