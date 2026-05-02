'use client';

import * as React from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { cn } from '@/lib/utils';

// ─── Types ────────────────────────────────────────────────────────────────────

interface BubbleColors {
    first?: string; second?: string; third?: string;
    fourth?: string; fifth?: string; sixth?: string;
}
interface SpringConfig { stiffness?: number; damping?: number }
export interface BubbleBackgroundProps extends React.HTMLAttributes<HTMLDivElement> {
    interactive?: boolean;
    colors?: BubbleColors;
    transition?: SpringConfig;
    containerRef?: React.RefObject<HTMLDivElement>;
}

// ─── Palette ─────────────────────────────────────────────────────────────────

const C = {
    red: { r: 220, g: 38, b: 38 },
    deepRed: { r: 153, g: 27, b: 27 },
    green: { r: 34, g: 197, b: 94 },
    blue: { r: 59, g: 130, b: 246 },
    cyan: { r: 34, g: 211, b: 238 },
    amber: { r: 245, g: 158, b: 11 },
    purple: { r: 168, g: 85, b: 247 },
    white: { r: 248, g: 250, b: 252 },
    bg: { r: 6, g: 6, b: 8 },
} as const;

type ColorRGB = { readonly r: number; readonly g: number; readonly b: number };
const rgba = (c: ColorRGB, a: number) => `rgba(${c.r},${c.g},${c.b},${a})`;

// ─── Single shared RAF coordinator ───────────────────────────────────────────

type DrawCallback = () => void;
class RAFCoordinator {
    private callbacks = new Set<DrawCallback>();
    private handle: number | null = null;
    private tick = () => {
        if (document.hidden) { this.handle = null; return; }
        this.callbacks.forEach(cb => cb());
        this.handle = requestAnimationFrame(this.tick);
    };
    register(cb: DrawCallback) {
        this.callbacks.add(cb);
        if (!this.handle && !document.hidden) this.handle = requestAnimationFrame(this.tick);
    }
    unregister(cb: DrawCallback) {
        this.callbacks.delete(cb);
        if (this.callbacks.size === 0 && this.handle !== null) {
            cancelAnimationFrame(this.handle); this.handle = null;
        }
    }
}

// ─── Shared canvas setup ──────────────────────────────────────────────────────

interface CanvasSetup {
    ctx: CanvasRenderingContext2D;
    getSize: () => { W: number; H: number };
    cleanup: () => void;
}

const setupCanvas = (canvas: HTMLCanvasElement, onResize?: () => void): CanvasSetup => {
    const ctx = canvas.getContext('2d')!;
    let W = 0, H = 0;
    const DPR = window.innerWidth < 768 ? 1 : Math.min(window.devicePixelRatio ?? 1, 2);

    const resize = (isSubsequent = false) => {
        W = canvas.offsetWidth; H = canvas.offsetHeight;
        canvas.width = W * DPR; canvas.height = H * DPR;
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
        if (isSubsequent) onResize?.();
    };
    resize(false);

    let resizeTimer: ReturnType<typeof setTimeout>;
    const ro = new ResizeObserver(() => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => resize(true), 100);
    });
    ro.observe(canvas);

    return {
        ctx,
        getSize: () => ({ W, H }),
        cleanup: () => { clearTimeout(resizeTimer); ro.disconnect(); },
    };
};

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 1 — CANDLESTICK + VOLUME + MULTI-SIGNAL CANVAS
// Enhanced: 3× brighter candles, volume bars, dual signal lines,
//           SELL labels, bigger pulse rings, depth grid
// ═══════════════════════════════════════════════════════════════════════════════

interface Candle {
    open: number; close: number; high: number; low: number;
    x: number; age: number; bullish: boolean; volume: number;
}
interface PulseRing { x: number; y: number; r: number; alpha: number; col: ColorRGB; thick: number }

const buildCandleDrawer = (canvas: HTMLCanvasElement): { draw: () => void; cleanup: () => void } => {
    const { ctx, getSize, cleanup } = setupCanvas(canvas);

    const COL_W = 24;
    const BODY_W = 9;
    const SPEED = 0.55;                       // ↑ faster scroll
    const PRICE_ZONE = { top: 0.18, bot: 0.70 };
    const VOL_ZONE = { top: 0.74, bot: 0.92 };
    const SCORE_LEN = 100;

    let price = 0.55;
    let candles: Candle[] = [];
    let pulseRings: PulseRing[] = [];
    let lastPulseX = -999;
    let t = 0, scoreTick = 0, scoreHead = 0;
    let flashAlpha = 0; // whole-canvas signal flash

    // Two independent oscillating price lines (alpha & momentum)
    const scoreRingA: number[] = Array.from({ length: SCORE_LEN },
        (_, i) => 0.40 + Math.sin(i * 0.19) * 0.10 + Math.sin(i * 0.07) * 0.06);
    const scoreRingB: number[] = Array.from({ length: SCORE_LEN },
        (_, i) => 0.55 + Math.sin(i * 0.13) * 0.08 + Math.cos(i * 0.09) * 0.05);

    const emitPulse = (x: number, y: number, bull: boolean, large = false) => {
        const col = bull ? C.green : C.red;
        pulseRings.push({ x, y, r: 3, alpha: large ? 1.0 : 0.75, col, thick: large ? 2.5 : 1.5 });
        if (large) pulseRings.push({ x, y, r: 8, alpha: 0.5, col, thick: 1 });
    };

    const makeCandle = (x: number): Candle => {
        const change = (Math.random() - 0.47) * 0.07;
        const open = price;
        const close = Math.max(0.08, Math.min(0.92, price + change));
        price = close;
        return {
            open, close,
            high: Math.max(open, close) + Math.random() * 0.030,
            low: Math.min(open, close) - Math.random() * 0.030,
            x, age: 0, bullish: close >= open,
            volume: 0.2 + Math.random() * 0.8,
        };
    };

    const pushScores = () => {
        const lastA = scoreRingA[scoreHead];
        scoreRingA[(scoreHead + 1) % SCORE_LEN] =
            Math.max(0.10, Math.min(0.90, lastA + (Math.random() - 0.48) * 0.022 + Math.sin(scoreTick * 0.04) * 0.007));
        const lastB = scoreRingB[scoreHead];
        scoreRingB[(scoreHead + 1) % SCORE_LEN] =
            Math.max(0.10, Math.min(0.90, lastB + (Math.random() - 0.50) * 0.018 + Math.cos(scoreTick * 0.035) * 0.005));
        scoreTick++;
        scoreHead = (scoreHead + 1) % SCORE_LEN;
    };

    const seed = (W: number) => {
        candles = [];
        const count = Math.ceil(W / COL_W) + 6;
        for (let i = 0; i < count; i++) candles.push(makeCandle(i * COL_W));
    };
    seed(800);

    // Floating metric labels (value slowly updates)
    const metrics = [
        { key: 'ALPHA', val: 74.2, col: C.green, x: 0.04, y: 0.09 },
        { key: 'TREND', val: 68.5, col: C.red, x: 0.72, y: 0.09 },
        { key: 'MOM', val: 61.3, col: C.amber, x: 0.86, y: 0.84 },
        { key: 'BETA', val: 0.82, col: C.blue, x: 0.04, y: 0.84 },
        { key: 'VOL', val: 22.1, col: C.purple, x: 0.44, y: 0.09 },
    ];

    // Bezier path helper (shared — avoids re-declaring inside draw)
    const drawBezierPath = (pts: Array<{ x: number; y: number }>) => {
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
            const p = pts[i - 1], c = pts[i], mx = (p.x + c.x) / 2;
            ctx.bezierCurveTo(mx, p.y, mx, c.y, c.x, c.y);
        }
    };

    const buildScorePts = (ring: number[], W: number, H: number) => {
        const pts: Array<{ x: number; y: number }> = [];
        const rLen = ring.length;
        for (let i = 0; i < rLen; i++) {
            const relIdx = (scoreHead - i + rLen) % rLen;
            const val = ring[relIdx];
            const sx = (candles.at(-1)?.x ?? W) - i * SPEED - i * (COL_W / (rLen / candles.length));
            const sy = H * (PRICE_ZONE.top + val * (PRICE_ZONE.bot - PRICE_ZONE.top));
            pts.unshift({ x: sx, y: sy });
        }
        return pts.filter(p => p.x >= -20 && p.x <= W + 20);
    };

    const draw = () => {
        const { W, H } = getSize();
        if (!W || !H) return;
        ctx.clearRect(0, 0, W, H);
        t += 0.009;

        // Spawn candles
        const rightmost = candles.at(-1);
        if (!rightmost || rightmost.x < W + COL_W) {
            candles.push(makeCandle((rightmost?.x ?? 0) + COL_W));
            pushScores();
        }
        candles.forEach(c => { c.x -= SPEED; c.age += 1; });
        candles = candles.filter(c => c.x > -COL_W * 2);

        // ── Depth perspective grid (NEW) ──────────────────────────────────────────
        // Subtle vanishing-point grid at the bottom gives depth and finance-terminal feel
        const VP = { x: W * 0.5, y: H * 0.68 };
        const GRID_LINES = 8;
        ctx.setLineDash([1, 6]);
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= GRID_LINES; i++) {
            const fx = (i / GRID_LINES) * W;
            ctx.beginPath();
            ctx.strokeStyle = rgba(C.blue, 0.04 + 0.02 * Math.abs(Math.sin(t + i * 0.3)));
            ctx.moveTo(fx, H);
            ctx.lineTo(VP.x, VP.y);
            ctx.stroke();
        }
        // Horizontal arcs in the grid zone
        for (let row = 1; row <= 5; row++) {
            const gy = VP.y + row * ((H - VP.y) / 5);
            ctx.beginPath();
            ctx.strokeStyle = rgba(C.blue, 0.025 * (1 - row / 6));
            ctx.moveTo(0, gy); ctx.lineTo(W, gy);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // ── Volume bars (NEW) ─────────────────────────────────────────────────────
        candles.forEach(c => {
            const fadeIn = Math.min(1, c.age / 20);
            const fadeOut = c.x < COL_W * 1.5 ? Math.max(0, c.x / (COL_W * 1.5)) : 1;
            const a = fadeIn * fadeOut * 0.28;
            const volH = (VOL_ZONE.bot - VOL_ZONE.top) * c.volume * H;
            const vy = H * VOL_ZONE.bot - volH;
            ctx.fillStyle = c.bullish ? rgba(C.green, a) : rgba(C.red, a);
            ctx.fillRect(c.x - BODY_W / 2, vy, BODY_W, volH);
        });

        // ── Candlesticks (ENHANCED — 3× brighter) ────────────────────────────────
        candles.forEach((c, idx) => {
            const fadeIn = Math.min(1, c.age / 25);
            const fadeOut = c.x < COL_W * 1.5 ? Math.max(0, c.x / (COL_W * 1.5)) : 1;
            // Base alpha raised from 0.14 → 0.38; oscillation tightened
            const alpha = fadeIn * fadeOut * (0.38 + 0.08 * Math.sin(t + idx * 0.7));

            const yHi = H * (PRICE_ZONE.top + c.high * (PRICE_ZONE.bot - PRICE_ZONE.top));
            const yLo = H * (PRICE_ZONE.top + c.low * (PRICE_ZONE.bot - PRICE_ZONE.top));
            const yOpen = H * (PRICE_ZONE.top + c.open * (PRICE_ZONE.bot - PRICE_ZONE.top));
            const yClos = H * (PRICE_ZONE.top + c.close * (PRICE_ZONE.bot - PRICE_ZONE.top));

            // Wick glow
            const col = c.bullish ? C.green : C.red;
            ctx.shadowColor = rgba(col, alpha * 0.6);
            ctx.shadowBlur = 4;
            ctx.beginPath();
            ctx.strokeStyle = rgba(col, alpha * 1.8);
            ctx.lineWidth = 1;
            ctx.moveTo(c.x, yHi); ctx.lineTo(c.x, yLo);
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Body
            const bodyTop = Math.min(yOpen, yClos);
            const bodyH = Math.max(Math.abs(yClos - yOpen), 2);
            ctx.fillStyle = rgba(col, alpha * 2.0);
            ctx.fillRect(c.x - BODY_W / 2, bodyTop, BODY_W, bodyH);

            // Bright top-edge highlight on bullish candles
            if (c.bullish && bodyH > 4) {
                ctx.fillStyle = rgba(C.white, alpha * 0.55);
                ctx.fillRect(c.x - BODY_W / 2, bodyTop, BODY_W, 2);
            }
        });

        // ── Price grid ────────────────────────────────────────────────────────────
        ctx.setLineDash([4, 8]);
        ctx.lineWidth = 0.75;
        for (let i = 1; i <= 5; i++) {
            const y = H * (PRICE_ZONE.top + (i / 6) * (PRICE_ZONE.bot - PRICE_ZONE.top));
            ctx.beginPath();
            ctx.strokeStyle = rgba(C.white, 0.045 + 0.01 * Math.sin(t * 0.5 + i));
            ctx.moveTo(0, y); ctx.lineTo(W, y);
            ctx.stroke();
            // Price tick labels on right edge
            const priceLbl = (1 - i / 6).toFixed(2);
            ctx.font = '500 8px "JetBrains Mono", monospace';
            ctx.fillStyle = rgba(C.white, 0.12);
            ctx.fillText(priceLbl, W - 34, y - 3);
        }
        ctx.setLineDash([]);

        // ── Signal line A (Alpha — red/crimson) ───────────────────────────────────
        const ptsA = buildScorePts(scoreRingA, W, H);
        if (ptsA.length > 2) {
            // Wide glow
            ctx.beginPath(); drawBezierPath(ptsA);
            ctx.strokeStyle = rgba(C.red, 0.22 + 0.07 * Math.sin(t * 1.2));
            ctx.lineWidth = 10;
            ctx.shadowColor = rgba(C.red, 0.5);
            ctx.shadowBlur = 20;
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Mid glow
            ctx.beginPath(); drawBezierPath(ptsA);
            ctx.strokeStyle = rgba(C.red, 0.55 + 0.1 * Math.sin(t * 0.9));
            ctx.lineWidth = 3;
            ctx.stroke();

            // Crisp core
            ctx.beginPath(); drawBezierPath(ptsA);
            ctx.strokeStyle = rgba(C.red, 0.88);
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Live dot + expanding ring
            const liveA = ptsA[ptsA.length - 1];
            const pulseA = (t * 2.5) % 1;
            ctx.beginPath(); ctx.arc(liveA.x, liveA.y, 4 + pulseA * 18, 0, Math.PI * 2);
            ctx.strokeStyle = rgba(C.red, 0.7 * (1 - pulseA));
            ctx.lineWidth = 1.5; ctx.stroke();
            ctx.beginPath(); ctx.arc(liveA.x, liveA.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = rgba(C.red, 1.0); ctx.fill();

            // BUY / SELL signal labels
            let lastCross = -999;
            for (let i = 1; i < ptsA.length; i++) {
                const prev = ptsA[i - 1], cur = ptsA[i];
                const THRESH_BUY = PRICE_ZONE.top + 0.44 * (PRICE_ZONE.bot - PRICE_ZONE.top);
                const THRESH_SELL = PRICE_ZONE.top + 0.30 * (PRICE_ZONE.bot - PRICE_ZONE.top);
                const py = prev.y / H, cy = cur.y / H;

                if (py > THRESH_BUY && cy < THRESH_BUY && cur.x - lastCross > 110) {
                    lastCross = cur.x;
                    const la = Math.min(1, (W - cur.x) / 80) * 0.75;
                    ctx.font = 'bold 9px "JetBrains Mono", monospace';
                    ctx.fillStyle = rgba(C.green, la);
                    ctx.fillText('▲ BUY', cur.x + 6, cur.y - 8);
                    if (cur.x - lastPulseX > 110) { emitPulse(cur.x, cur.y, true, true); lastPulseX = cur.x; }
                }
                if (py < THRESH_SELL && cy > THRESH_SELL && cur.x - lastCross > 110) {
                    lastCross = cur.x;
                    const la = Math.min(1, (W - cur.x) / 80) * 0.65;
                    ctx.font = 'bold 9px "JetBrains Mono", monospace';
                    ctx.fillStyle = rgba(C.red, la);
                    ctx.fillText('▼ SELL', cur.x + 6, cur.y + 14);
                    if (cur.x - lastPulseX > 110) { emitPulse(cur.x, cur.y, false, true); lastPulseX = cur.x; }
                }
            }
        }

        // ── Signal line B (Momentum — cyan/teal) ──────────────────────────────────
        const ptsB = buildScorePts(scoreRingB, W, H);
        if (ptsB.length > 2) {
            ctx.beginPath(); drawBezierPath(ptsB);
            ctx.strokeStyle = rgba(C.cyan, 0.14 + 0.05 * Math.sin(t * 0.8 + 1));
            ctx.lineWidth = 7;
            ctx.shadowColor = rgba(C.cyan, 0.35);
            ctx.shadowBlur = 14;
            ctx.stroke();
            ctx.shadowBlur = 0;

            ctx.beginPath(); drawBezierPath(ptsB);
            ctx.strokeStyle = rgba(C.cyan, 0.65 + 0.1 * Math.sin(t * 1.1));
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Dashed cyan live dot
            const liveB = ptsB[ptsB.length - 1];
            const pulseB = (t * 2.0 + 0.5) % 1;
            ctx.beginPath(); ctx.arc(liveB.x, liveB.y, 3 + pulseB * 14, 0, Math.PI * 2);
            ctx.strokeStyle = rgba(C.cyan, 0.6 * (1 - pulseB));
            ctx.lineWidth = 1; ctx.stroke();
            ctx.beginPath(); ctx.arc(liveB.x, liveB.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = rgba(C.cyan, 0.9); ctx.fill();
        }

        // ── Crosshair on live candle (NEW) ────────────────────────────────────────
        const liveCandle = candles.at(-1);
        if (liveCandle) {
            const lcy = H * (PRICE_ZONE.top + liveCandle.close * (PRICE_ZONE.bot - PRICE_ZONE.top));
            ctx.setLineDash([3, 5]);
            ctx.beginPath();
            ctx.strokeStyle = rgba(C.white, 0.08 + 0.04 * Math.sin(t * 3));
            ctx.lineWidth = 0.5;
            ctx.moveTo(0, lcy); ctx.lineTo(W, lcy);
            ctx.stroke();
            // Right-edge price tag
            ctx.setLineDash([]);
            ctx.fillStyle = rgba(C.white, 0.15);
            ctx.fillRect(W - 42, lcy - 7, 40, 13);
            ctx.fillStyle = rgba(C.white, 0.65);
            ctx.font = 'bold 8px "JetBrains Mono", monospace';
            ctx.fillText(liveCandle.close.toFixed(3), W - 40, lcy + 4);
        }

        // ── Floating metric labels (ENHANCED — larger, brighter) ─────────────────
        ctx.font = 'bold 9px "JetBrains Mono", monospace';
        metrics.forEach((l, i) => {
            // Slowly drift values
            l.val += (Math.random() - 0.5) * 0.02;
            const a = 0.45 + 0.18 * Math.sin(t * 0.6 + i * 1.3);
            ctx.fillStyle = rgba(l.col, a);
            ctx.fillText(`${l.key} ${l.val.toFixed(2)}`, W * l.x, H * l.y);
            // Subtle underline glow
            const tw = ctx.measureText(`${l.key} ${l.val.toFixed(2)}`).width;
            ctx.beginPath();
            ctx.strokeStyle = rgba(l.col, a * 0.4);
            ctx.lineWidth = 0.5;
            ctx.moveTo(W * l.x, H * l.y + 2);
            ctx.lineTo(W * l.x + tw, H * l.y + 2);
            ctx.stroke();
        });

        // ── Pulse rings (ENHANCED — faster expand, triple-ring on large) ─────────
        pulseRings = pulseRings.filter(p => p.alpha > 0.01);
        pulseRings.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.strokeStyle = rgba(p.col, p.alpha);
            ctx.lineWidth = p.thick;
            ctx.shadowColor = rgba(p.col, p.alpha * 0.5);
            ctx.shadowBlur = 6;
            ctx.stroke();
            ctx.shadowBlur = 0;
            p.r += 1.4;
            p.alpha *= 0.92;
        });

        // ── Whole-canvas signal flash (NEW) ───────────────────────────────────────
        if (flashAlpha > 0.001) {
            ctx.fillStyle = rgba(C.green, flashAlpha * 0.07);
            ctx.fillRect(0, 0, W, H);
            flashAlpha *= 0.85;
        }
        if (Math.random() < 0.003) flashAlpha = 1.0;
    };

    return { draw, cleanup };
};

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 2 — DATA RAIN CANVAS
// Enhanced: 2× more drops, brighter head, wider chars, colour-shift over time
// ═══════════════════════════════════════════════════════════════════════════════

const DATA_CHARS = '0123456789.+-×÷%↑↓←→◆▲▼●NIFTYBTCETHUSD₹$€¥∑∏∂∇≈≠∞'.split('');

interface RainDrop {
    x: number; y: number; speed: number; length: number;
    chars: string[]; col: ColorRGB; alpha: number; fontSize: number;
}

const buildRainDrawer = (canvas: HTMLCanvasElement): { draw: () => void; cleanup: () => void } => {
    const { ctx, getSize, cleanup } = setupCanvas(canvas);
    const RAIN_COLS: ColorRGB[] = [C.green, C.red, C.blue, C.cyan, C.amber, C.purple];

    const randChar = () => DATA_CHARS[Math.floor(Math.random() * DATA_CHARS.length)];
    const randColIdx = () => Math.floor(Math.random() * RAIN_COLS.length);

    const spawnDrop = (): RainDrop => {
        const { W } = getSize();
        const fontSize = 8 + Math.floor(Math.random() * 5); // varied sizes: 8–12px
        return {
            x: Math.random() * (W || 800),
            y: -Math.random() * 300,
            speed: 0.5 + Math.random() * 1.0,               // ↑ faster
            length: 8 + Math.floor(Math.random() * 16),       // ↑ longer streams
            chars: Array.from({ length: 22 }, randChar),
            col: RAIN_COLS[randColIdx()],
            alpha: 0.09 + Math.random() * 0.16,              // ↑ much brighter
            fontSize,
        };
    };

    // ↑ 60 drops instead of 30
    let drops: RainDrop[] = Array.from({ length: 60 }, spawnDrop);

    const draw = () => {
        const { W, H } = getSize();
        if (!W || !H) return;
        ctx.clearRect(0, 0, W, H);

        // Spawn extras more aggressively
        if (Math.random() < 0.08 && drops.length < 80) drops.push(spawnDrop());

        drops.forEach((d, di) => {
            d.y += d.speed;
            if (Math.random() < 0.08) d.chars[Math.floor(Math.random() * d.chars.length)] = randChar();

            const LINE_H = d.fontSize + 2;
            ctx.font = `${d.fontSize}px "JetBrains Mono", monospace`;

            for (let i = 0; i < d.length; i++) {
                const charY = d.y - i * LINE_H;
                if (charY < -10 || charY > H + 10) continue;

                const progress = 1 - i / d.length;
                let headAlpha: number;
                let headCol: ColorRGB;

                if (i === 0) {
                    // Head: bright white with colour tint
                    headAlpha = d.alpha * 5.0;
                    headCol = C.white;
                } else if (i <= 2) {
                    // Near-head: colour at near-full brightness
                    headAlpha = d.alpha * 3.5 * progress;
                    headCol = d.col;
                } else {
                    headAlpha = d.alpha * progress;
                    headCol = d.col;
                }

                ctx.fillStyle = rgba(headCol, Math.min(1, headAlpha));
                ctx.fillText(d.chars[i % d.chars.length], d.x, charY);
            }

            if (d.y - d.length * (d.fontSize + 2) > H + 30) drops[di] = spawnDrop();
        });
    };

    return { draw, cleanup };
};

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 3 — CONSTELLATION CANVAS
// Enhanced: 30 nodes, bigger radii, brighter edges, "warp" lines on close pairs,
//           node size pulses more dramatically, star-burst on near-collision
// ═══════════════════════════════════════════════════════════════════════════════

interface ParticleNode {
    x: number; y: number; vx: number; vy: number;
    r: number; col: ColorRGB; alpha: number;
    pulse: number; pulseSpeed: number;
    trail: Array<{ x: number; y: number }>;
}

const buildConstellationDrawer = (canvas: HTMLCanvasElement): { draw: () => void; cleanup: () => void } => {
    let nodes: ParticleNode[] = [];
    const NODE_COLS: ColorRGB[] = [C.blue, C.cyan, C.green, C.amber, C.red, C.purple];

    const spawnNode = (W: number, H: number): ParticleNode => ({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.28,  // ↑ faster drift
        vy: (Math.random() - 0.5) * 0.22,
        r: 1.5 + Math.random() * 2.5,      // ↑ bigger nodes
        col: NODE_COLS[Math.floor(Math.random() * NODE_COLS.length)],
        alpha: 0.35 + Math.random() * 0.45, // ↑ much more visible
        pulse: Math.random() * Math.PI * 2,
        pulseSpeed: 0.016 + Math.random() * 0.024,
        trail: [],
    });

    const { ctx, getSize, cleanup } = setupCanvas(canvas, () => {
        const W = canvas.offsetWidth || 800, H = canvas.offsetHeight || 600;
        nodes = Array.from({ length: 30 }, () => spawnNode(W, H));
    });

    const { W: W0, H: H0 } = getSize();
    nodes = Array.from({ length: 30 }, () => spawnNode(W0 || 800, H0 || 600));

    const CONNECT_DIST = 140;         // ↑ wider connection range
    const CONNECT_DIST_SQ = CONNECT_DIST * CONNECT_DIST;
    const WARP_DIST_SQ = 55 * 55;    // special bright edge when very close

    let t = 0;

    const draw = () => {
        const { W, H } = getSize();
        if (!W || !H) return;
        ctx.clearRect(0, 0, W, H);
        t += 0.012;

        nodes.forEach(n => {
            n.x += n.vx;
            n.y += n.vy;
            n.pulse += n.pulseSpeed;
            if (n.x < 0 || n.x > W) n.vx *= -1;
            if (n.y < 0 || n.y > H) n.vy *= -1;
            n.x = Math.max(0, Math.min(W, n.x));
            n.y = Math.max(0, Math.min(H, n.y));

            // Trail (last 6 positions)
            n.trail.push({ x: n.x, y: n.y });
            if (n.trail.length > 6) n.trail.shift();
        });

        // ── Edges ─────────────────────────────────────────────────────────────────
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                const dx = a.x - b.x, dy = a.y - b.y;
                const dSq = dx * dx + dy * dy;
                if (dSq > CONNECT_DIST_SQ) continue;

                const dist = Math.sqrt(dSq);
                const strength = 1 - dist / CONNECT_DIST;
                const isWarp = dSq < WARP_DIST_SQ;

                if (isWarp) {
                    // Bright "warp" line between very close nodes
                    ctx.beginPath();
                    ctx.strokeStyle = rgba(C.white, strength * 0.55);
                    ctx.lineWidth = 1.5;
                    ctx.shadowColor = rgba(a.col, 0.5);
                    ctx.shadowBlur = 8;
                    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                } else {
                    // Normal edge — brighter than before (0.07 → 0.18)
                    ctx.beginPath();
                    ctx.strokeStyle = rgba(C.blue, strength * 0.18);
                    ctx.lineWidth = strength * 0.8;
                    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }

        // ── Node trails ───────────────────────────────────────────────────────────
        nodes.forEach(n => {
            if (n.trail.length < 2) return;
            for (let i = 1; i < n.trail.length; i++) {
                const a = n.trail[i - 1], b = n.trail[i];
                const ta = (i / n.trail.length) * 0.15;
                ctx.beginPath();
                ctx.strokeStyle = rgba(n.col, ta);
                ctx.lineWidth = 0.5;
                ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
                ctx.stroke();
            }
        });

        // ── Nodes ─────────────────────────────────────────────────────────────────
        nodes.forEach(n => {
            const pulsedAlpha = n.alpha * (0.6 + 0.4 * Math.sin(n.pulse));       // ↑ more range
            const pulsedR = n.r * (0.8 + 0.2 * Math.sin(n.pulse * 1.4));  // ↑ more size variance

            // Outer halo — wide shadow glow
            ctx.shadowColor = rgba(n.col, pulsedAlpha * 0.7);
            ctx.shadowBlur = pulsedR * 9;
            ctx.beginPath();
            ctx.arc(n.x, n.y, pulsedR, 0, Math.PI * 2);
            ctx.fillStyle = rgba(n.col, pulsedAlpha);
            ctx.fill();
            ctx.shadowBlur = 0;

            // Bright core dot
            ctx.beginPath();
            ctx.arc(n.x, n.y, pulsedR * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = rgba(C.white, pulsedAlpha * 0.7);
            ctx.fill();
        });
    };

    return { draw, cleanup };
};

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 4 — TICKER STRIPS (DOM)
// Enhanced: 4 strips, varied opacity & direction, stronger colours
// ═══════════════════════════════════════════════════════════════════════════════

const TICKER_DATA = [
    { sym: 'NIFTY50', val: '24,312.15', chg: '+0.82%', up: true },
    { sym: 'SENSEX', val: '80,248.07', chg: '+0.91%', up: true },
    { sym: 'RELIANCE', val: '2,841.30', chg: '-0.34%', up: false },
    { sym: 'TCS', val: '4,102.65', chg: '+1.24%', up: true },
    { sym: 'HDFCBANK', val: '1,752.20', chg: '-0.17%', up: false },
    { sym: 'INFY', val: '1,524.80', chg: '+0.76%', up: true },
    { sym: 'ICICIBANK', val: '1,194.45', chg: '+0.53%', up: true },
    { sym: 'SBIN', val: '822.95', chg: '-0.45%', up: false },
    { sym: 'TATAMTR', val: '967.30', chg: '+1.62%', up: true },
    { sym: 'WIPRO', val: '558.10', chg: '-0.28%', up: false },
    { sym: 'BTC/USD', val: '93,204', chg: '+2.14%', up: true },
    { sym: 'ETH/USD', val: '3,418.90', chg: '+1.87%', up: true },
    { sym: 'GOLD', val: '73,842', chg: '+0.61%', up: true },
    { sym: 'CRUDE', val: '6,724', chg: '-0.33%', up: false },
    { sym: 'BAJFIN', val: '7,241.30', chg: '+1.14%', up: true },
];

// Pre-computed outside component
const TICKERS_FWD = [...TICKER_DATA, ...TICKER_DATA];
const TICKERS_REV = [...TICKER_DATA, ...TICKER_DATA];

const TickerStrip: React.FC<{
    disabled: boolean; speed?: number; reverse?: boolean; opacity?: number; top?: string;
}> = React.memo(({ disabled, speed = 22, reverse = false, opacity = 0.35, top = '0px' }) => {
    if (disabled) return null;
    const items = reverse ? TICKERS_REV : TICKERS_FWD;
    return (
        <div
            className="absolute left-0 right-0 overflow-hidden pointer-events-none z-[1]"
            style={{ top, opacity }}
            aria-hidden="true"
        >
            <motion.div
                className="flex gap-5 whitespace-nowrap"
                initial={{ x: reverse ? '-50%' : '0%' }}
                animate={{ x: reverse ? '0%' : '-50%' }}
                transition={{ duration: speed, repeat: Infinity, ease: 'linear' }}
            >
                {items.map((item, i) => (
                    <span key={i} className="inline-flex items-center gap-1.5 font-mono text-[9px] tracking-widest">
                        <span style={{ color: rgba(C.white, 0.55) }}>{item.sym}</span>
                        <span style={{ color: rgba(C.white, 0.35) }}>{item.val}</span>
                        <span style={{ color: item.up ? rgba(C.green, 0.9) : rgba(C.red, 0.9) }}>{item.chg}</span>
                        <span style={{ color: rgba(C.white, 0.12) }}>│</span>
                    </span>
                ))}
            </motion.div>
        </div>
    );
});
TickerStrip.displayName = 'TickerStrip';

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 5 — HEAT MAP GRID
// Enhanced: brighter colours, bigger cells, faster refresh
// ═══════════════════════════════════════════════════════════════════════════════

const HEAT_TICKERS = [
    { sym: 'RELIANCE' }, { sym: 'TCS' }, { sym: 'HDFCBANK' }, { sym: 'INFY' },
    { sym: 'ICICIBANK' }, { sym: 'WIPRO' }, { sym: 'SBIN' }, { sym: 'BAJFIN' },
    { sym: 'TATAMTR' }, { sym: 'SUNPHRM' }, { sym: 'ADANIENT' }, { sym: 'LT' },
    { sym: 'MARUTI' }, { sym: 'NTPC' }, { sym: 'POWGRID' }, { sym: 'ASIANPNT' },
];

interface HeatCell { sym: string; score: number; delta: number }

const scoreToColor = (s: number) =>
    s < 35 ? rgba(C.red, 0.14 + (1 - s / 35) * 0.16)
        : s < 55 ? rgba(C.amber, 0.08 + ((s - 35) / 20) * 0.10)
            : rgba(C.green, 0.10 + ((s - 55) / 45) * 0.20);

const scoreToBorder = (s: number) =>
    s < 35 ? rgba(C.red, 0.22 + (1 - s / 35) * 0.18)
        : s < 55 ? rgba(C.amber, 0.14)
            : rgba(C.green, 0.16 + ((s - 55) / 45) * 0.20);

const HeatMapGrid: React.FC<{ disabled: boolean }> = React.memo(({ disabled }) => {
    const [cells, setCells] = React.useState<HeatCell[]>(() =>
        HEAT_TICKERS.map(t => ({ ...t, score: 30 + Math.random() * 55, delta: (Math.random() - 0.5) * 0.5 })),
    );

    React.useEffect(() => {
        if (disabled) return;
        // ↑ refresh every 900ms instead of 1200ms — more lively
        const id = setInterval(() => {
            setCells(prev => prev.map(cell => {
                const newDelta = cell.delta * 0.88 + (Math.random() - 0.5) * 0.35;
                return { ...cell, score: Math.max(8, Math.min(96, cell.score + newDelta)), delta: newDelta };
            }));
        }, 900);
        return () => clearInterval(id);
    }, [disabled]);

    if (disabled) return null;

    const Cell = ({ cell }: { cell: HeatCell }) => (
        <motion.div
            key={cell.sym}
            className="flex flex-col items-center justify-center rounded-[3px] border px-[7px] py-[5px] min-w-[56px]"
            animate={{ backgroundColor: scoreToColor(cell.score), borderColor: scoreToBorder(cell.score) }}
            transition={{ duration: 0.85, ease: 'easeInOut' }}
        >
            <span className="text-[8px] font-mono font-bold tracking-wider leading-none"
                style={{ color: cell.score < 35 ? rgba(C.red, 0.80) : cell.score < 55 ? rgba(C.amber, 0.80) : rgba(C.green, 0.80) }}>
                {cell.sym}
            </span>
            <motion.span className="text-[7px] font-mono leading-none mt-0.5"
                animate={{ color: cell.score < 35 ? rgba(C.red, 0.60) : cell.score < 55 ? rgba(C.amber, 0.60) : rgba(C.green, 0.60) }}
                transition={{ duration: 0.85 }}>
                {cell.score.toFixed(1)}
            </motion.span>
        </motion.div>
    );

    return (
        <div className="absolute inset-0 z-0 pointer-events-none" aria-hidden="true">
            <div className="absolute top-0 left-0 right-0 h-[40%] flex items-start pt-2 px-2 gap-[3px] flex-wrap content-start overflow-hidden opacity-90">
                {cells.slice(0, 8).map(cell => <Cell key={cell.sym} cell={cell} />)}
            </div>
            <div className="absolute bottom-0 left-0 right-0 h-[32%] flex items-end pb-6 px-2 gap-[3px] flex-wrap-reverse content-end overflow-hidden opacity-80">
                {cells.slice(8).map(cell => <Cell key={cell.sym} cell={cell} />)}
            </div>
        </div>
    );
});
HeatMapGrid.displayName = 'HeatMapGrid';

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 6 — ORBITAL RINGS (CSS, compositor thread)
// Enhanced: more rings, higher opacity, varied widths
// ═══════════════════════════════════════════════════════════════════════════════

const orbitalStyle = `
@keyframes spin-cw  { from { transform: rotate(0deg)   } to { transform: rotate(360deg)  } }
@keyframes spin-ccw { from { transform: rotate(0deg)   } to { transform: rotate(-360deg) } }
`;

const OrbitalRings: React.FC<{ disabled: boolean }> = React.memo(({ disabled }) => {
    if (disabled) return null;
    return (
        <>
            <style>{orbitalStyle}</style>
            <div className="absolute inset-0 pointer-events-none z-[1] overflow-hidden" aria-hidden="true">
                {/* Ring 1 — large slow CW blue */}
                <div style={{
                    position: 'absolute', width: '150%', height: '85%', left: '-25%', top: '8%',
                    border: `1px solid ${rgba(C.blue, 0.12)}`, borderRadius: '50%',
                    animation: 'spin-cw 90s linear infinite',
                }} />
                {/* Ring 2 — tall CCW green */}
                <div style={{
                    position: 'absolute', width: '105%', height: '145%', left: '-2%', top: '-22%',
                    border: `1.5px solid ${rgba(C.green, 0.09)}`, borderRadius: '50%',
                    animation: 'spin-ccw 70s linear infinite',
                }} />
                {/* Ring 3 — small fast CW red, bottom-right */}
                <div style={{
                    position: 'absolute', width: '55%', height: '65%', right: '-12%', bottom: '-12%',
                    border: `1px solid ${rgba(C.red, 0.12)}`, borderRadius: '50%',
                    animation: 'spin-cw 38s linear infinite',
                }} />
                {/* Ring 4 — dashed amber centre */}
                <div style={{
                    position: 'absolute', width: '72%', height: '92%', left: '14%', top: '4%',
                    border: `1px dashed ${rgba(C.amber, 0.08)}`, borderRadius: '50%',
                    animation: 'spin-cw 120s linear infinite',
                }} />
                {/* Ring 5 — NEW: thin cyan top-left accent */}
                <div style={{
                    position: 'absolute', width: '35%', height: '50%', left: '-8%', top: '-10%',
                    border: `1px solid ${rgba(C.cyan, 0.08)}`, borderRadius: '50%',
                    animation: 'spin-ccw 55s linear infinite',
                }} />
                {/* Ring 6 — NEW: wide ellipse across centre */}
                <div style={{
                    position: 'absolute', width: '120%', height: '40%', left: '-10%', top: '30%',
                    border: `0.5px solid ${rgba(C.purple, 0.07)}`, borderRadius: '50%',
                    animation: 'spin-cw 150s linear infinite',
                }} />
            </div>
        </>
    );
});
OrbitalRings.displayName = 'OrbitalRings';

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 7 — CORNER HUD BRACKETS
// Enhanced: larger size, thicker stroke, additional inner crosshair marks
// ═══════════════════════════════════════════════════════════════════════════════

const CornerBrackets: React.FC<{ disabled: boolean }> = React.memo(({ disabled }) => {
    if (disabled) return null;
    const SZ = 28; // ↑ larger than 20
    const b = (style: React.CSSProperties) => (
        <div style={{
            position: 'absolute', width: SZ, height: SZ,
            border: `1.5px solid ${rgba(C.red, 0.35)}`,
            borderRadius: 1, ...style,
        }} />
    );
    return (
        <motion.div
            className="absolute inset-0 pointer-events-none z-[4]"
            animate={{ opacity: [0.45, 1, 0.45] }}
            transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
            aria-hidden="true"
        >
            {b({ top: 10, left: 10, borderRight: 'none', borderBottom: 'none' })}
            {b({ top: 10, right: 10, borderLeft: 'none', borderBottom: 'none' })}
            {b({ bottom: 10, left: 10, borderRight: 'none', borderTop: 'none' })}
            {b({ bottom: 10, right: 10, borderLeft: 'none', borderTop: 'none' })}
            {/* Inner tick marks — NEW */}
            {[
                { top: 10 + SZ / 2 - 2, left: 10 },
                { top: 10, left: 10 + SZ / 2 - 2 },
                { top: 10 + SZ / 2 - 2, right: 10 },
                { top: 10, right: 10 + SZ / 2 - 2 },
                { bottom: 10 + SZ / 2 - 2, left: 10 },
                { bottom: 10, left: 10 + SZ / 2 - 2 },
                { bottom: 10 + SZ / 2 - 2, right: 10 },
                { bottom: 10, right: 10 + SZ / 2 - 2 },
            ].map((s, i) => (
                <div key={i} style={{
                    position: 'absolute', width: 4, height: 4,
                    background: rgba(C.red, 0.5), borderRadius: '50%', ...s,
                }} />
            ))}
        </motion.div>
    );
});
CornerBrackets.displayName = 'CornerBrackets';

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 8 — AMBIENT GLOW
// Enhanced: doubled intensity, 2 extra glows, tighter breath timing
// ═══════════════════════════════════════════════════════════════════════════════

const AmbientGlow: React.FC<{ disabled: boolean }> = React.memo(({ disabled }) => {
    if (disabled) return null;
    return (
        <>
            {/* Central dark void */}
            <motion.div
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-[1] w-[650px] h-[430px] rounded-full pointer-events-none"
                style={{ background: 'radial-gradient(ellipse at center, rgba(6,6,8,0.65) 0%, rgba(6,6,8,0.0) 72%)' }}
                animate={{ scale: [1, 1.08, 1], opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
                aria-hidden="true"
            />
            {/* Colour orbs — significantly boosted alpha */}
            {[
                { col: C.red, pos: 'bottom-[10%] left-[8%]', sz: 'w-[380px] h-[260px]', blur: 'blur-[70px]', a: 0.14, delay: 1.5, dur: 8 },
                { col: C.green, pos: 'top-[8%] right-[6%]', sz: 'w-[320px] h-[240px]', blur: 'blur-[65px]', a: 0.10, delay: 2.5, dur: 10 },
                { col: C.blue, pos: 'top-[35%] right-[3%]', sz: 'w-[240px] h-[340px]', blur: 'blur-[55px]', a: 0.11, delay: 2, dur: 7.5 },
                { col: C.cyan, pos: 'top-[5%] left-[5%]', sz: 'w-[220px] h-[190px]', blur: 'blur-[50px]', a: 0.09, delay: 4, dur: 12 },
                { col: C.amber, pos: 'bottom-[5%] left-[38%]', sz: 'w-[300px] h-[160px]', blur: 'blur-[60px]', a: 0.08, delay: 5.5, dur: 9 },
                { col: C.purple, pos: 'top-[40%] left-[2%]', sz: 'w-[200px] h-[280px]', blur: 'blur-[65px]', a: 0.07, delay: 3, dur: 11 },
                { col: C.red, pos: 'top-[15%] left-[35%]', sz: 'w-[180px] h-[180px]', blur: 'blur-[80px]', a: 0.06, delay: 7, dur: 14 },
            ].map(({ col, pos, sz, blur, a, delay, dur }, i) => (
                <motion.div
                    key={i}
                    className={`absolute ${pos} z-[1] ${sz} rounded-full pointer-events-none ${blur}`}
                    style={{ background: rgba(col, a) }}
                    animate={{ scale: [1, 1.22, 1], opacity: [0.5, 1.0, 0.5] }}
                    transition={{ duration: dur, repeat: Infinity, ease: 'easeInOut', delay }}
                    aria-hidden="true"
                />
            ))}
        </>
    );
});
AmbientGlow.displayName = 'AmbientGlow';

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM 9 — RADAR SWEEP (NEW DOM system)
// A rotating conic-gradient sweep suggesting market scanning
// ═══════════════════════════════════════════════════════════════════════════════

const radarStyle = `
@keyframes radar-spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
@keyframes radar-fade { 0%,100% { opacity:0.35 } 50% { opacity:0.7 } }
`;

const RadarSweep: React.FC<{ disabled: boolean }> = React.memo(({ disabled }) => {
    if (disabled) return null;
    return (
        <>
            <style>{radarStyle}</style>
            <div
                className="absolute inset-0 pointer-events-none z-[1] overflow-hidden"
                aria-hidden="true"
            >
                <div style={{
                    position: 'absolute',
                    width: '140%', height: '140%',
                    top: '-20%', left: '-20%',
                    background: 'conic-gradient(from 0deg, transparent 0deg, rgba(220,38,38,0.04) 20deg, transparent 40deg)',
                    animation: 'radar-spin 12s linear infinite, radar-fade 6s ease-in-out infinite',
                    borderRadius: '50%',
                }} />
                {/* Second slower sweep — cyan */}
                <div style={{
                    position: 'absolute',
                    width: '120%', height: '120%',
                    top: '-10%', left: '-10%',
                    background: 'conic-gradient(from 90deg, transparent 0deg, rgba(34,211,238,0.03) 15deg, transparent 30deg)',
                    animation: 'radar-spin 20s linear infinite reverse',
                    borderRadius: '50%',
                }} />
            </div>
        </>
    );
});
RadarSweep.displayName = 'RadarSweep';

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT — BubbleBackground
// ═══════════════════════════════════════════════════════════════════════════════

export const BubbleBackground = React.forwardRef<HTMLDivElement, BubbleBackgroundProps>(
    ({ className, interactive = true, colors = {}, transition = { stiffness: 60, damping: 20 }, containerRef, ...props }, ref) => {
        const prefersReduced = useReducedMotion();
        const disabled = !!prefersReduced;

        const candleRef = React.useRef<HTMLCanvasElement>(null);
        const rainRef = React.useRef<HTMLCanvasElement>(null);
        const constellRef = React.useRef<HTMLCanvasElement>(null);

        // Single shared RAF loop drives all three canvases
        React.useEffect(() => {
            if (disabled) return;
            if (!candleRef.current || !rainRef.current || !constellRef.current) return;

            const candle = buildCandleDrawer(candleRef.current);
            const rain = buildRainDrawer(rainRef.current);
            const constell = buildConstellationDrawer(constellRef.current);
            const coord = new RAFCoordinator();

            const drawAll = () => {
                constell.draw();
                rain.draw();
                candle.draw();
            };

            coord.register(drawAll);
            const onVis = () => {
                if (document.hidden) coord.unregister(drawAll);
                else coord.register(drawAll);
            };
            document.addEventListener('visibilitychange', onVis);

            return () => {
                coord.unregister(drawAll);
                document.removeEventListener('visibilitychange', onVis);
                candle.cleanup(); rain.cleanup(); constell.cleanup();
            };
        }, [disabled]);

        return (
            <div
                ref={ref}
                className={cn('relative overflow-hidden bg-transparent', className)}
                {...props}
            >
                {/* Layer 0 — Constellation (deepest) */}
                {!disabled && (
                    <canvas ref={constellRef} style={{ willChange: 'transform' }}
                        className="absolute inset-0 w-full h-full z-[0] pointer-events-none" aria-hidden="true" />
                )}

                {/* Layer 1 — Heat Map Grid */}
                <HeatMapGrid disabled={disabled || !interactive} />

                {/* Layer 2 — Orbital Rings */}
                <OrbitalRings disabled={disabled} />

                {/* Layer 2b — Radar Sweep (new) */}
                <RadarSweep disabled={disabled} />

                {/* Layer 3 — Data Rain */}
                {!disabled && (
                    <canvas ref={rainRef} style={{ willChange: 'transform' }}
                        className="absolute inset-0 w-full h-full z-[2] pointer-events-none" aria-hidden="true" />
                )}

                {/* Layer 4 — Candlestick + Score lines */}
                {!disabled && (
                    <canvas ref={candleRef} style={{ willChange: 'transform' }}
                        className="absolute inset-0 w-full h-full z-[3] pointer-events-none" aria-hidden="true" />
                )}

                {/* Layer 5 — Ticker strips (4 rows, varied opacity & direction) */}
                <TickerStrip disabled={disabled} speed={22} reverse={false} opacity={0.55} top="44%" />
                <TickerStrip disabled={disabled} speed={14} reverse={true} opacity={0.35} top="50%" />
                <TickerStrip disabled={disabled} speed={34} reverse={false} opacity={0.22} top="56%" />
                <TickerStrip disabled={disabled} speed={44} reverse={true} opacity={0.14} top="62%" />

                {/* Layer 6 — Ambient glow */}
                <AmbientGlow disabled={disabled} />

                {/* Layer 7 — Scan lines (3 axes, CSS + motion) */}
                {!disabled && (
                    <>
                        <motion.div
                            className="absolute left-0 right-0 h-[2px] z-[5] pointer-events-none"
                            style={{ background: `linear-gradient(to right, transparent, ${rgba(C.red, 0.30)} 25%, ${rgba(C.red, 0.30)} 75%, transparent)` }}
                            initial={{ top: '0%', opacity: 0 }}
                            animate={{ top: ['0%', '100%'], opacity: [0, 0.9, 0] }}
                            transition={{ duration: 6, repeat: Infinity, ease: 'linear', repeatDelay: 8 }}
                            aria-hidden="true"
                        />
                        <motion.div
                            className="absolute left-0 right-0 h-px z-[5] pointer-events-none"
                            style={{ background: `linear-gradient(to right, transparent, ${rgba(C.cyan, 0.18)} 35%, ${rgba(C.cyan, 0.18)} 65%, transparent)` }}
                            initial={{ top: '100%', opacity: 0 }}
                            animate={{ top: ['100%', '0%'], opacity: [0, 0.6, 0] }}
                            transition={{ duration: 4.5, repeat: Infinity, ease: 'linear', repeatDelay: 12, delay: 3.5 }}
                            aria-hidden="true"
                        />
                        <motion.div
                            className="absolute top-0 bottom-0 w-[2px] z-[5] pointer-events-none"
                            style={{ background: `linear-gradient(to bottom, transparent, ${rgba(C.blue, 0.18)} 30%, ${rgba(C.blue, 0.18)} 70%, transparent)` }}
                            initial={{ left: '0%', opacity: 0 }}
                            animate={{ left: ['0%', '100%'], opacity: [0, 0.55, 0] }}
                            transition={{ duration: 8, repeat: Infinity, ease: 'linear', repeatDelay: 16, delay: 7 }}
                            aria-hidden="true"
                        />
                        {/* Short fast red flash-scan — NEW */}
                        <motion.div
                            className="absolute left-0 right-0 h-[3px] z-[5] pointer-events-none"
                            style={{ background: `linear-gradient(to right, transparent 0%, ${rgba(C.green, 0.35)} 45%, ${rgba(C.green, 0.35)} 55%, transparent)` }}
                            initial={{ top: '50%', opacity: 0 }}
                            animate={{ top: ['30%', '70%', '30%'], opacity: [0, 0.7, 0.7, 0] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', repeatDelay: 18, delay: 11 }}
                            aria-hidden="true"
                        />
                    </>
                )}

                {/* Layer 8 — Corner HUD brackets */}
                <CornerBrackets disabled={disabled} />
            </div>
        );
    },
);
BubbleBackground.displayName = 'BubbleBackground';