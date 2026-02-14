/**
 * Main thread API for communicating with the ChessKnot engine Web Worker
 */

const _v = new URL(import.meta.url).searchParams.get('v') || '';
const _q = _v ? `?v=${_v}` : '';

export class EngineAPI {
    constructor() {
        this.worker = null;
        this.pendingRequests = new Map();
        this.requestId = 0;
        this.isReady = false;
        this.onReady = null;
        this.onError = null;
    }

    async init(distBase) {
        const workerUrl = new URL(`./engine-worker.js${_q}`, import.meta.url).href;
        return new Promise((resolve, reject) => {
            this.worker = new Worker(workerUrl);

            this.worker.onmessage = (e) => {
                const { id, type, ...data } = e.data;

                if (type === 'ready') {
                    this.isReady = true;
                    if (this.onReady) this.onReady();
                    resolve();
                    return;
                }

                if (type === 'error') {
                    if (this.onError) this.onError(data.message);
                    reject(new Error(data.message));
                    return;
                }

                if (id !== undefined && this.pendingRequests.has(id)) {
                    const { resolve, reject } = this.pendingRequests.get(id);
                    this.pendingRequests.delete(id);

                    if (data.error) {
                        reject(new Error(data.error));
                    } else {
                        resolve(data);
                    }
                }
            };

            this.worker.onerror = (err) => {
                if (this.onError) this.onError(err.message);
                reject(err);
            };

            // Tell worker where to find the engine WASM
            this.worker.postMessage({ type: 'init', data: { distBase } });
        });
    }

    async request(type, data = {}) {
        if (!this.worker) throw new Error('Worker not initialized');

        return new Promise((resolve, reject) => {
            const id = ++this.requestId;
            this.pendingRequests.set(id, { resolve, reject });
            this.worker.postMessage({ id, type, data });
        });
    }

    async loadNNModel(data) {
        return this.request('loadNNModel', { data });
    }

    async getLegalMoves() {
        const response = await this.request('getLegalMoves');
        return response.moves;
    }

    async makeMove(from, to) {
        const response = await this.request('makeMove', { from, to });
        return response.board;
    }

    async resetBoard() {
        const response = await this.request('resetBoard');
        return response.board;
    }

    async getBoard() {
        const response = await this.request('getBoard');
        return response.board;
    }

    async getLockedPieces() {
        const response = await this.request('getLockedPieces');
        return response.locked;
    }

    async search(softTime = 3, hardTime = 10) {
        const response = await this.request('search', { softTime, hardTime });
        return response.result;
    }

    stopSearch() {
        // Without SharedArrayBuffer, we can't interrupt the search mid-execution.
        // The search will stop naturally when TimeControl's hard time limit is reached.
    }

    async getStatus() {
        return this.request('getStatus');
    }

    terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
            this.isReady = false;
        }
    }
}

let engineInstance = null;

export function getEngine() {
    if (!engineInstance) {
        engineInstance = new EngineAPI();
    }
    return engineInstance;
}
