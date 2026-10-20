import math
import time
from typing import Dict, Any, List
import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Time-Series Visualization Research Workbench",
    description="НИР: Сравнительный анализ Client-Side Raw Data vs Server-Side LTTB Reduction",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# МАТЕМАТИЧЕСКОЕ ЯДРО: ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ И АЛГОРИТМ LTTB (NUMPY)
# ==============================================================================

def generate_signal(count: int, signal_type: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Векторизованный генератор синтетических временных рядов с экстремумами.
    Возвращает кортеж (x, y, время_генерации_мс).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 100.0, count, dtype=np.float64)

    if signal_type == "sine_noise":
        # Полигармонический сигнал + гауссовский шум + редкие импульсные выбросы (спайки)
        y = (
                10.0 * np.sin(2.0 * np.pi * 0.08 * x)
                + 4.5 * np.cos(2.0 * np.pi * 0.35 * x)
                + 2.0 * np.sin(2.0 * np.pi * 1.2 * x)
                + rng.normal(0.0, 1.4, size=count)
        )
        # Добавляем 15 детерминированных острых пиков для проверки сохранения экстремумов
        spike_indices = np.linspace(int(count * 0.05), int(count * 0.95), 15, dtype=np.int64)
        spike_signs = np.array([1 if i % 2 == 0 else -1 for i in range(15)], dtype=np.float64)
        y[spike_indices] += spike_signs * 18.0

    elif signal_type == "random_walk":
        # Броуновское движение (дискретный винеровский процесс) с возвратным дрейфом
        steps = rng.normal(0.0, 1.0, size=count)
        raw_walk = np.cumsum(steps) / math.sqrt(count / 1000.0)
        envelope = 8.0 * np.sin(2.0 * np.pi * 0.03 * x)
        y = raw_walk + envelope

    elif signal_type == "damped_harmonic":
        # Затухающие гармонические колебания с серией повторных импульсных возбуждений
        y = np.zeros(count, dtype=np.float64)
        bursts = [0.0, 28.0, 58.0, 82.0]
        amps = [25.0, -20.0, 18.0, -15.0]
        for b_time, amp in zip(bursts, amps):
            mask = x >= b_time
            t_local = x[mask] - b_time
            y[mask] += amp * np.exp(-0.14 * t_local) * np.cos(2.0 * np.pi * 0.65 * t_local)
        y += rng.normal(0.0, 0.45, size=count)

    else:
        y = np.sin(x) + rng.normal(0.0, 0.2, size=count)

    t_gen_ms = (time.perf_counter() - t0) * 1000.0
    return x, y, t_gen_ms


def lttb_downsample(x: np.ndarray, y: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Высокоскоростная реализация алгоритма LTTB (Largest-Triangle-Three-Buckets) на NumPy.
    Предвычисляет центры тяжести корзин через np.add.reduceat.
    """
    t0 = time.perf_counter()
    n = x.shape[0]

    if threshold >= n or threshold < 3:
        return x.copy(), y.copy(), (time.perf_counter() - t0) * 1000.0

    sampled_x = np.empty(threshold, dtype=np.float64)
    sampled_y = np.empty(threshold, dtype=np.float64)

    # Крайние точки всегда сохраняются
    sampled_x[0] = x[0]
    sampled_y[0] = y[0]
    sampled_x[-1] = x[-1]
    sampled_y[-1] = y[-1]

    every = (n - 2) / (threshold - 2)

    # Векторизованный расчет границ корзин
    idx = np.arange(threshold - 1, dtype=np.float64)
    boundaries = np.floor(idx * every).astype(np.int64) + 1
    boundaries = np.append(boundaries, n)

    # Предвычисление средних координат (x_C, y_C) для следующих корзин за O(N) в C-коде
    next_starts = boundaries[1:-1]
    counts = np.diff(boundaries[1:])
    avg_x_all = np.add.reduceat(x, next_starts) / counts
    avg_y_all = np.add.reduceat(y, next_starts) / counts

    a = 0  # Индекс последней выбранной точки A
    for i in range(threshold - 2):
        b_start = boundaries[i]
        b_end = boundaries[i + 1]

        avg_x = avg_x_all[i]
        avg_y = avg_y_all[i]

        ax = x[a]
        ay = y[a]

        bx = x[b_start:b_end]
        by = y[b_start:b_end]

        # Векторизованное вычисление удвоенной площади треугольников в корзине
        areas = np.abs((ax - avg_x) * (by - ay) - (ax - bx) * (avg_y - ay))
        max_offset = int(np.argmax(areas))

        a = b_start + max_offset
        sampled_x[i + 1] = x[a]
        sampled_y[i + 1] = y[a]

    t_reduce_ms = (time.perf_counter() - t0) * 1000.0
    return sampled_x, sampled_y, t_reduce_ms


def compute_approximation_metrics(
        x_raw: np.ndarray, y_raw: np.ndarray, x_red: np.ndarray, y_red: np.ndarray
) -> Dict[str, float]:
    """
    Оценивает математическую точность редукции:
    - RMSE (среднеквадратичная ошибка линейной интерполяции на исходной сетке)
    - R^2 (доля сохраненной дисперсии / информативности формы)
    - Peak Retention (%) (точность сохранения глобального размаха амплитуды)
    """
    y_interp = np.interp(x_raw, x_red, y_red)
    residuals = y_raw - y_interp
    mse = float(np.mean(residuals ** 2))
    rmse = math.sqrt(mse)

    ss_tot = float(np.sum((y_raw - np.mean(y_raw)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    raw_ptp = float(np.ptp(y_raw))
    red_ptp = float(np.ptp(y_red))
    peak_retention = (red_ptp / raw_ptp * 100.0) if raw_ptp > 0 else 100.0

    return {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "peak_retention_pct": round(min(100.0, peak_retention), 2),
    }


def serialize_points(x: np.ndarray, y: np.ndarray) -> List[Dict[str, float]]:
    """Формирует список словарей {'x': float, 'y': float} для передачи по JSON."""
    x_list = np.round(x, 5).tolist()
    y_list = np.round(y, 4).tolist()
    return [{"x": xv, "y": yv} for xv, yv in zip(x_list, y_list)]


# ==============================================================================
# REST API ЭНДПОИНТЫ
# ==============================================================================

@app.get("/api/generate/raw")
def api_generate_raw(
        count: int = Query(10000, ge=100, le=500000),
        signal_type: str = Query("sine_noise"),
):
    t_start = time.perf_counter()
    x, y, t_gen_ms = generate_signal(count, signal_type)

    t_ser_0 = time.perf_counter()
    points = serialize_points(x, y)
    t_ser_ms = (time.perf_counter() - t_ser_0) * 1000.0
    t_total_server_ms = (time.perf_counter() - t_start) * 1000.0

    return JSONResponse(
        content={
            "mode": "raw",
            "meta": {
                "raw_count": count,
                "returned_count": count,
                "signal_type": signal_type,
                "server_gen_ms": round(t_gen_ms, 3),
                "server_reduce_ms": 0.0,
                "server_serialize_ms": round(t_ser_ms, 3),
                "server_total_ms": round(t_total_server_ms, 3),
                "compression_ratio": 1.0,
                "rmse": 0.0,
                "r2": 1.0,
                "peak_retention_pct": 100.0,
            },
            "data": points,
        }
    )


@app.get("/api/generate/downsampled")
def api_generate_downsampled(
        count: int = Query(10000, ge=100, le=500000),
        threshold: int = Query(750, ge=50, le=5000),
        signal_type: str = Query("sine_noise"),
):
    t_start = time.perf_counter()
    x, y, t_gen_ms = generate_signal(count, signal_type)
    rx, ry, t_reduce_ms = lttb_downsample(x, y, threshold)

    t_ser_0 = time.perf_counter()
    points = serialize_points(rx, ry)
    t_ser_ms = (time.perf_counter() - t_ser_0) * 1000.0
    t_total_server_ms = (time.perf_counter() - t_start) * 1000.0

    # Вычисляем метрики точности вне основного таймера сериализации
    fidelity = compute_approximation_metrics(x, y, rx, ry)

    return JSONResponse(
        content={
            "mode": "lttb",
            "meta": {
                "raw_count": count,
                "returned_count": len(points),
                "threshold": threshold,
                "signal_type": signal_type,
                "server_gen_ms": round(t_gen_ms, 3),
                "server_reduce_ms": round(t_reduce_ms, 3),
                "server_serialize_ms": round(t_ser_ms, 3),
                "server_total_ms": round(t_total_server_ms, 3),
                "compression_ratio": round(count / max(1, len(points)), 1),
                **fidelity,
            },
            "data": points,
        }
    )


@app.get("/api/benchmark-suite")
def api_benchmark_suite(
        threshold: int = Query(750, ge=50, le=5000),
        signal_type: str = Query("sine_noise"),
):
    """
    Серверный пакетный бенчмарк для быстрой сверки математических показателей
    по всей сетке N = [1000, 5000, 20000, 50000, 100000, 250000].
    """
    sizes = [1000, 5000, 20000, 50000, 100000, 250000]
    results = []

    for n in sizes:
        # 1. Замер Raw
        t0 = time.perf_counter()
        x, y, t_gen = generate_signal(n, signal_type)
        raw_pts = serialize_points(x, y)
        raw_json_str = str(raw_pts)
        t_raw_server = (time.perf_counter() - t0) * 1000.0

        # 2. Замер LTTB
        t1 = time.perf_counter()
        x2, y2, t_gen2 = generate_signal(n, signal_type)
        rx, ry, t_lttb = lttb_downsample(x2, y2, threshold)
        lttb_pts = serialize_points(rx, ry)
        lttb_json_str = str(lttb_pts)
        t_lttb_server = (time.perf_counter() - t1) * 1000.0

        fidelity = compute_approximation_metrics(x2, y2, rx, ry)

        results.append({
            "n": n,
            "k": len(lttb_pts),
            "raw_server_ms": round(t_raw_server, 2),
            "lttb_server_ms": round(t_lttb_server, 2),
            "lttb_algo_only_ms": round(t_lttb, 2),
            "raw_payload_kb": round(len(raw_json_str.encode("utf-8")) / 1024.0, 1),
            "lttb_payload_kb": round(len(lttb_json_str.encode("utf-8")) / 1024.0, 1),
            **fidelity,
        })

    return JSONResponse(content={"suite": results, "threshold": threshold, "signal_type": signal_type})


# ==============================================================================
# КЛИЕНТСКИЙ ИНТЕРФЕЙС (HTML5 / TAILWIND CSS / CHART.JS / TELEMETRY HUD)
# ==============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>НИР: Визуализация больших временных рядов | Raw vs LTTB</title>

    <!-- Шрифты Inter и JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        slate: {
                            950: '#060911',
                            900: '#0b101e',
                            850: '#111827',
                            800: '#1e293b',
                        }
                    }
                }
            }
        }
    </script>

    <!-- Chart.js & Lucide Icons -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>

    <style>
        body {
            background-color: #060911;
            color: #e2e8f0;
            overflow-x: hidden;
        }
        .glass-panel {
            background: rgba(11, 16, 30, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(51, 65, 85, 0.55);
        }
        .glow-indigo {
            box-shadow: 0 0 25px -5px rgba(99, 102, 241, 0.25);
        }
        .glow-emerald {
            box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.25);
        }
        .glow-rose {
            box-shadow: 0 0 25px -5px rgba(244, 63, 94, 0.25);
        }
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #060911;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
    </style>
</head>
<body class="min-h-screen flex flex-col font-sans antialiased selection:bg-indigo-500 selection:text-white">

    <!-- ВЕРХНЯЯ ПАНЕЛЬ НАВИГАЦИИ И МОНИТОРИНГА ПОТОКА БРАУЗЕРА -->
    <header class="sticky top-0 z-30 glass-panel border-b border-slate-800/80 px-6 py-3">
        <div class="max-w-[1800px] mx-auto flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-3">
                <div class="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/40 flex items-center justify-center text-indigo-400">
                    <i data-lucide="activity" class="w-5 h-5"></i>
                </div>
                <div>
                    <div class="flex items-center gap-2">
                        <span class="text-xs font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">НИР • ПМиИ</span>
                        <h1 class="text-sm md:text-base font-semibold text-white tracking-tight">
                            Исследовательский стенд: Client-Side Raw vs Server-Side LTTB
                        </h1>
                    </div>
                    <p class="text-xs text-slate-400 hidden sm:block">
                        Анализ пропускной способности, десериализации V8 и задержек растеризации HTML5 Canvas на выборках до 250 000 точек
                    </p>
                </div>
            </div>

            <!-- Индикаторы состояния Main Thread и FPS в реальном времени -->
            <div class="flex items-center gap-4">
                <div class="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
                    <div class="flex items-center gap-2">
                        <span id="fpsDot" class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span class="text-xs text-slate-400">UI Thread:</span>
                        <span id="fpsValue" class="text-xs font-mono font-semibold text-emerald-400">60 FPS</span>
                    </div>
                    <div class="h-3 w-[1px] bg-slate-800"></div>
                    <div class="flex items-center gap-1.5" title="Максимальная блокировка основного потока браузера за последний запрос">
                        <span class="text-xs text-slate-400">Max Frame Lag:</span>
                        <span id="lagValue" class="text-xs font-mono font-semibold text-slate-200">16.7 мс</span>
                    </div>
                </div>

                <div id="statusBadge" class="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-mono">
                    <i data-lucide="check-circle-2" class="w-3.5 h-3.5"></i>
                    <span id="statusText">Ядро NumPy готово</span>
                </div>
            </div>
        </div>
    </header>

    <!-- ОСНОВНАЯ РАБОЧАЯ ОБЛАСТЬ -->
    <div class="flex-1 max-w-[1800px] w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-5 p-5">

        <!-- ЛЕВАЯ ПАНЕЛЬ УПРАВЛЕНИЯ (SIDEBAR) -->
        <aside class="lg:col-span-3 flex flex-col gap-4">

            <!-- Блок 1: Конфигурация генератора и выборки -->
            <div class="glass-panel rounded-xl p-4 flex flex-col gap-4">
                <div class="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                    <div class="flex items-center gap-2">
                        <i data-lucide="sliders" class="w-4 h-4 text-indigo-400"></i>
                        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-200">Параметры выборки</h2>
                    </div>
                    <span id="selectedNBadge" class="text-xs font-mono text-indigo-400 font-semibold">N = 50 000</span>
                </div>

                <!-- Выбор объема выборки N -->
                <div>
                    <label class="text-xs text-slate-400 block mb-2">Объем исходного ряда (точек N):</label>
                    <div class="grid grid-cols-3 gap-1.5" id="nButtonsContainer">
                        <button onclick="selectN(1000)" data-n="1000" class="n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 transition">1 000</button>
                        <button onclick="selectN(5000)" data-n="5000" class="n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 transition">5 000</button>
                        <button onclick="selectN(20000)" data-n="20000" class="n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 transition">20 000</button>
                        <button onclick="selectN(50000)" data-n="50000" class="n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-indigo-500 bg-indigo-500/20 text-white font-semibold transition">50 000</button>
                        <button onclick="selectN(100000)" data-n="100000" class="n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 transition">100 000</button>
                        <button onclick="selectN(250000)" data-n="250000" class="n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 transition">250 000</button>
                    </div>
                </div>

                <!-- Выбор окна редукции K -->
                <div>
                    <div class="flex justify-between text-xs mb-1.5">
                        <span class="text-slate-400">Окно редукции LTTB (порог K):</span>
                        <span id="thresholdLabel" class="font-mono text-emerald-400 font-semibold">750 точек</span>
                    </div>
                    <input id="thresholdSlider" type="range" min="200" max="2000" step="50" value="750"
                        oninput="updateThreshold(this.value)"
                        class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">
                    <div class="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                        <span>200 (Макс. сжатие)</span>
                        <span>750 (Оптимум)</span>
                        <span>2000 (Детально)</span>
                    </div>
                </div>

                <!-- Выбор математической модели сигнала -->
                <div>
                    <label class="text-xs text-slate-400 block mb-1.5">Математическая модель сигнала:</label>
                    <select id="signalSelect" onchange="updateSignalFormula()"
                        class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
                        <option value="sine_noise">Полигармонический + Гауссовский шум + Выбросы</option>
                        <option value="random_walk">Броуновское движение (Random Walk)</option>
                        <option value="damped_harmonic">Затухающий осциллятор с импульсами</option>
                    </select>
                    <div id="formulaBox" class="mt-2 p-2.5 rounded-lg bg-slate-950/90 border border-slate-800/80 font-mono text-[11px] text-slate-400">
                        x(t) = Σ Aᵢ·sin(ωᵢt) + N(0, σ²) + δ(t - tₖ)
                    </div>
                </div>
            </div>

            <!-- Блок 2: Запуск экспериментов -->
            <div class="glass-panel rounded-xl p-4 flex flex-col gap-3">
                <div class="flex items-center gap-2 border-b border-slate-800/80 pb-2.5">
                    <i data-lucide="cpu" class="w-4 h-4 text-emerald-400"></i>
                    <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-200">Запуск экспериментов</h2>
                </div>

                <!-- Кнопка 1: Тяжелый клиент (Raw) -->
                <button id="btnRunRaw" onclick="runSingleExperiment('raw')"
                    class="w-full group relative flex items-center justify-between px-4 py-3 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 hover:border-rose-500/60 transition text-left">
                    <div>
                        <div class="text-xs font-semibold text-rose-300 flex items-center gap-1.5">
                            <i data-lucide="database" class="w-3.5 h-3.5"></i>
                            <span>1. Запрос Raw (Client-Side)</span>
                        </div>
                        <p class="text-[11px] text-slate-400 mt-0.5">Передача и рендер всех N точек без сжатия</p>
                    </div>
                    <i data-lucide="chevron-right" class="w-4 h-4 text-rose-400 group-hover:translate-x-0.5 transition"></i>
                </button>

                <!-- Кнопка 2: Серверная редукция (LTTB) -->
                <button id="btnRunLttb" onclick="runSingleExperiment('lttb')"
                    class="w-full group relative flex items-center justify-between px-4 py-3 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 hover:border-emerald-500/60 transition text-left">
                    <div>
                        <div class="text-xs font-semibold text-emerald-300 flex items-center gap-1.5">
                            <i data-lucide="zap" class="w-3.5 h-3.5"></i>
                            <span>2. Запрос LTTB (Server-Side)</span>
                        </div>
                        <p class="text-[11px] text-slate-400 mt-0.5">Векторизованная редукция до K точек на сервере</p>
                    </div>
                    <i data-lucide="chevron-right" class="w-4 h-4 text-emerald-400 group-hover:translate-x-0.5 transition"></i>
                </button>

                <!-- Кнопка 3: Режим наложения (Overlay Verification) -->
                <button id="btnRunOverlay" onclick="runOverlayComparison()"
                    class="w-full group relative flex items-center justify-between px-4 py-3 rounded-xl bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 hover:border-indigo-500/60 transition text-left">
                    <div>
                        <div class="text-xs font-semibold text-indigo-300 flex items-center gap-1.5">
                            <i data-lucide="layers" class="w-3.5 h-3.5"></i>
                            <span>3. Наложение Raw + LTTB</span>
                        </div>
                        <p class="text-[11px] text-slate-400 mt-0.5">Визуальная верификация сохранения экстремумов</p>
                    </div>
                    <i data-lucide="chevron-right" class="w-4 h-4 text-indigo-400 group-hover:translate-x-0.5 transition"></i>
                </button>

                <div class="my-1 border-t border-slate-800/80"></div>

                <!-- Кнопка 4: Автоматический полный стресс-тест -->
                <button id="btnStressTest" onclick="runFullBenchmarkSuite()"
                    class="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-emerald-600 hover:from-indigo-500 hover:to-emerald-500 text-white font-semibold text-xs shadow-lg shadow-indigo-500/20 flex items-center justify-center gap-2 transition">
                    <i data-lucide="play" class="w-4 h-4 fill-current"></i>
                    <span>Автоматический полный стресс-тест</span>
                </button>

                <!-- Прогресс-бар стресс-теста -->
                <div id="suiteProgressContainer" class="hidden flex-col gap-1.5 pt-1">
                    <div class="flex justify-between text-[11px] font-mono">
                        <span id="suiteProgressLabel" class="text-indigo-300">Выполнение...</span>
                        <span id="suiteProgressPct" class="text-slate-400">0%</span>
                    </div>
                    <div class="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                        <div id="suiteProgressBar" class="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <!-- Блок 3: Справка по методике измерения -->
            <div class="glass-panel rounded-xl p-4 text-xs text-slate-400 space-y-2">
                <div class="flex items-center gap-2 text-slate-200 font-semibold">
                    <i data-lucide="microscope" class="w-4 h-4 text-indigo-400"></i>
                    <span>Методика телеметрии</span>
                </div>
                <p class="leading-relaxed text-[11px]">
                    Декомпозиция задержки фиксируется независимо на четырех участках тракта:
                </p>
                <ul class="space-y-1 text-[11px] font-mono text-slate-300">
                    <li>• <span class="text-indigo-400">T_srv</span>: Генерация + редукция NumPy</li>
                    <li>• <span class="text-sky-400">T_net</span>: Передача HTTP-потока</li>
                    <li>• <span class="text-amber-400">T_prs</span>: Десериализация JSON.parse()</li>
                    <li>• <span class="text-emerald-400">T_rnd</span>: Растеризация линий Canvas</li>
                </ul>
            </div>
        </aside>

        <!-- ПРАВАЯ ОБЛАСТЬ: HUD МЕТРИК, ОСЦИЛЛОГРАФ, АНАЛИТИКА И ТАБЛИЦА -->
        <main class="lg:col-span-9 flex flex-col gap-5">

            <!-- РЯД 1: TELEMETRY HUD (6 КАРТОЧЕК ВЫСОКОГО РАЗРЕШЕНИЯ) -->
            <section class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">

                <!-- 1. Server Time -->
                <div class="glass-panel rounded-xl p-3.5 border-l-2 border-l-indigo-500">
                    <div class="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Server Time</span>
                        <i data-lucide="server" class="w-3.5 h-3.5 text-indigo-400"></i>
                    </div>
                    <div id="hudServerTime" class="text-lg font-mono font-bold text-white mt-1">— мс</div>
                    <div id="hudServerSub" class="text-[10px] font-mono text-slate-400 mt-1 truncate">Ожидание запуска</div>
                </div>

                <!-- 2. Payload Size -->
                <div class="glass-panel rounded-xl p-3.5 border-l-2 border-l-sky-500">
                    <div class="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Payload Size</span>
                        <i data-lucide="hard-drive-download" class="w-3.5 h-3.5 text-sky-400"></i>
                    </div>
                    <div id="hudPayloadSize" class="text-lg font-mono font-bold text-white mt-1">— КБ</div>
                    <div id="hudPayloadSub" class="text-[10px] font-mono text-slate-400 mt-1 truncate">Сжатие: 1.0x</div>
                </div>

                <!-- 3. Network + Parse -->
                <div class="glass-panel rounded-xl p-3.5 border-l-2 border-l-amber-500">
                    <div class="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Net + JSON Parse</span>
                        <i data-lucide="network" class="w-3.5 h-3.5 text-amber-400"></i>
                    </div>
                    <div id="hudNetParseTime" class="text-lg font-mono font-bold text-white mt-1">— мс</div>
                    <div id="hudNetParseSub" class="text-[10px] font-mono text-slate-400 mt-1 truncate">Net: — | Parse: —</div>
                </div>

                <!-- 4. Client Canvas Render -->
                <div class="glass-panel rounded-xl p-3.5 border-l-2 border-l-rose-500">
                    <div class="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Client Render</span>
                        <i data-lucide="monitor" class="w-3.5 h-3.5 text-rose-400"></i>
                    </div>
                    <div id="hudRenderTime" class="text-lg font-mono font-bold text-white mt-1">— мс</div>
                    <div id="hudRenderSub" class="text-[10px] font-mono text-slate-400 mt-1 truncate">HTML5 Canvas 2D</div>
                </div>

                <!-- 5. Total Pipeline Latency -->
                <div class="glass-panel rounded-xl p-3.5 border-l-2 border-l-purple-500">
                    <div class="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Total Latency</span>
                        <i data-lucide="timer" class="w-3.5 h-3.5 text-purple-400"></i>
                    </div>
                    <div id="hudTotalTime" class="text-lg font-mono font-bold text-white mt-1">— мс</div>
                    <div id="hudTotalSub" class="text-[10px] font-mono text-slate-400 mt-1 truncate">Полный цикл E2E</div>
                </div>

                <!-- 6. Speedup & Accuracy -->
                <div class="glass-panel rounded-xl p-3.5 border-l-2 border-l-emerald-500 bg-emerald-500/5">
                    <div class="flex items-center justify-between text-[11px] text-emerald-300">
                        <span>Speedup / R²</span>
                        <i data-lucide="gauge" class="w-3.5 h-3.5 text-emerald-400"></i>
                    </div>
                    <div id="hudSpeedup" class="text-lg font-mono font-bold text-emerald-400 mt-1">—</div>
                    <div id="hudFidelitySub" class="text-[10px] font-mono text-slate-300 mt-1 truncate">Пики: 100% | R²: 1.00</div>
                </div>
            </section>

            <!-- РЯД 2: ГЛАВНЫЙ ИНТЕРАКТИВНЫЙ ГРАФИК ВРЕМЕННОГО РЯДА -->
            <section class="glass-panel rounded-xl p-4 flex flex-col gap-3">
                <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                    <div class="flex items-center gap-2.5">
                        <span class="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                        <h3 class="text-sm font-semibold text-white">Осциллограмма временного ряда</h3>
                        <span id="chartModeTag" class="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                            Режим: Ожидание
                        </span>
                    </div>

                    <div class="flex items-center gap-3 text-xs">
                        <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
                            <input type="checkbox" id="togglePoints" onchange="togglePointMarkers()" class="rounded bg-slate-900 border-slate-700 text-indigo-500">
                            <span>Показывать узлы LTTB</span>
                        </label>
                        <span id="renderedCountInfo" class="font-mono text-slate-400 text-[11px]">Отрисовано: 0 точек</span>
                    </div>
                </div>

                <div class="relative w-full h-[360px]">
                    <canvas id="mainSeriesChart"></canvas>
                </div>
            </section>

            <!-- РЯД 3: АНАЛИТИЧЕСКИЕ ГРАФИКИ МАСШТАБИРУЕМОСТИ И ДЕКОМПОЗИЦИИ -->
            <section class="grid grid-cols-1 md:grid-cols-2 gap-5">

                <!-- График масштабируемости T(N) -->
                <div class="glass-panel rounded-xl p-4 flex flex-col gap-3">
                    <div class="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                        <div>
                            <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-200">
                                Кривая масштабируемости T(N)
                            </h3>
                            <p class="text-[11px] text-slate-400">Зависимость полной задержки тракта (мс) от объема выборки N</p>
                        </div>
                        <i data-lucide="trending-up" class="w-4 h-4 text-indigo-400"></i>
                    </div>
                    <div class="relative w-full h-[250px]">
                        <canvas id="scalabilityChart"></canvas>
                    </div>
                </div>

                <!-- График декомпозиции затрат по фазам -->
                <div class="glass-panel rounded-xl p-4 flex flex-col gap-3">
                    <div class="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                        <div>
                            <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-200">
                                Профиль фазовых задержек (мс)
                            </h3>
                            <p class="text-[11px] text-slate-400">Сравнение вклада Server, Network/Parse и Canvas Render</p>
                        </div>
                        <i data-lucide="bar-chart-3" class="w-4 h-4 text-emerald-400"></i>
                    </div>
                    <div class="relative w-full h-[250px]">
                        <canvas id="breakdownChart"></canvas>
                    </div>
                </div>
            </section>

            <!-- РЯД 4: СВОДНАЯ НАУЧНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ И ЭКСПОРТ В CSV / LATEX -->
            <section class="glass-panel rounded-xl p-4 flex flex-col gap-3">
                <div class="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                    <div>
                        <h3 class="text-sm font-semibold text-white flex items-center gap-2">
                            <i data-lucide="table" class="w-4 h-4 text-indigo-400"></i>
                            <span>Протокол экспериментальных измерений (Raw Data vs LTTB Reduction)</span>
                        </h3>
                        <p class="text-xs text-slate-400 mt-0.5">
                            Данные автоматически агрегируются при одиночных запусках и во время полного стресс-теста
                        </p>
                    </div>

                    <div class="flex items-center gap-2">
                        <button onclick="exportTableToCSV()"
                            class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-mono flex items-center gap-1.5 transition">
                            <i data-lucide="download" class="w-3.5 h-3.5 text-emerald-400"></i>
                            <span>Экспорт CSV</span>
                        </button>
                        <button onclick="copyLatexTable()"
                            class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-mono flex items-center gap-1.5 transition">
                            <i data-lucide="code-2" class="w-3.5 h-3.5 text-indigo-400"></i>
                            <span>Копировать LaTeX</span>
                        </button>
                        <button onclick="clearBenchmarkTable()"
                            class="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-rose-950/50 text-slate-400 hover:text-rose-300 border border-slate-800 text-xs transition"
                            title="Очистить таблицу">
                            <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                        </button>
                    </div>
                </div>

                <div class="overflow-x-auto">
                    <table class="w-full text-left border-collapse text-xs font-mono">
                        <thead>
                            <tr class="border-b border-slate-800 text-slate-400 bg-slate-900/50">
                                <th class="py-2.5 px-3">Выборка N</th>
                                <th class="py-2.5 px-3">Окно K</th>
                                <th class="py-2.5 px-3 text-rose-300">Raw Payload</th>
                                <th class="py-2.5 px-3 text-emerald-300">LTTB Payload</th>
                                <th class="py-2.5 px-3 text-rose-300">Raw Render</th>
                                <th class="py-2.5 px-3 text-emerald-300">LTTB Render</th>
                                <th class="py-2.5 px-3 text-rose-300">Raw Total (мс)</th>
                                <th class="py-2.5 px-3 text-emerald-300">LTTB Total (мс)</th>
                                <th class="py-2.5 px-3 text-indigo-300">Speedup</th>
                                <th class="py-2.5 px-3">Пики (%) / RMSE</th>
                            </tr>
                        </thead>
                        <tbody id="benchmarkTableBody" class="divide-y divide-slate-800/60 text-slate-200">
                            <tr>
                                <td colspan="10" class="py-6 text-center text-slate-500 font-sans">
                                    Нажмите «Автоматический полный стресс-тест» или выполните парные запросы для заполнения таблицы
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

        </main>
    </div>

    <!-- СКРИПТ УПРАВЛЕНИЯ СТЕНДОМ, ТЕЛЕМЕТРИЕЙ И ГРАФИКАМИ -->
    <script>
        lucide.createIcons();

        // Глобальное состояние стенда
        const state = {
            selectedN: 50000,
            thresholdK: 750,
            signalType: 'sine_noise',
            showPointMarkers: false,
            lastRawRun: {},   // Хранит результаты по ключам N
            lastLttbRun: {},  // Хранит результаты по ключам N
            tableRecords: {}  // Сводный словарь по N для таблицы и графиков
        };

        // ----------------------------------------------------------------------
        // 1. МОНИТОРИНГ FPS И БЛОКИРОВКИ MAIN THREAD (REQUESTANIMATIONFRAME)
        // ----------------------------------------------------------------------
        let frameCount = 0;
        let lastFpsTime = performance.now();
        let lastFrameStamp = performance.now();
        let maxFrameGapMs = 16.7;
        let trackingLagWindow = false;

        function monitorMainThread(now) {
            const delta = now - lastFrameStamp;
            lastFrameStamp = now;
            if (trackingLagWindow && delta > maxFrameGapMs) {
                maxFrameGapMs = delta;
            }

            frameCount++;
            if (now - lastFpsTime >= 500) {
                const fps = Math.min(60, Math.round((frameCount * 1000) / (now - lastFpsTime)));
                const fpsEl = document.getElementById('fpsValue');
                const dotEl = document.getElementById('fpsDot');
                const lagEl = document.getElementById('lagValue');

                fpsEl.textContent = `${fps} FPS`;
                lagEl.textContent = `${maxFrameGapMs.toFixed(1)} мс`;

                if (fps >= 50) {
                    fpsEl.className = 'text-xs font-mono font-semibold text-emerald-400';
                    dotEl.className = 'w-2 h-2 rounded-full bg-emerald-400';
                } else if (fps >= 25) {
                    fpsEl.className = 'text-xs font-mono font-semibold text-amber-400';
                    dotEl.className = 'w-2 h-2 rounded-full bg-amber-400';
                } else {
                    fpsEl.className = 'text-xs font-mono font-semibold text-rose-400';
                    dotEl.className = 'w-2 h-2 rounded-full bg-rose-500 animate-ping';
                }

                if (maxFrameGapMs > 100) {
                    lagEl.className = 'text-xs font-mono font-semibold text-rose-400';
                } else if (maxFrameGapMs > 35) {
                    lagEl.className = 'text-xs font-mono font-semibold text-amber-400';
                } else {
                    lagEl.className = 'text-xs font-mono font-semibold text-slate-200';
                }

                frameCount = 0;
                lastFpsTime = now;
            }
            requestAnimationFrame(monitorMainThread);
        }
        requestAnimationFrame(monitorMainThread);

        // ----------------------------------------------------------------------
        // 2. ИНИЦИАЛИЗАЦИЯ ГРАФИКОВ CHART.JS
        // ----------------------------------------------------------------------
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'JetBrains Mono', monospace";

        const mainCtx = document.getElementById('mainSeriesChart').getContext('2d');
        const mainChart = new Chart(mainCtx, {
            type: 'line',
            data: { datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false, // Прямая передача {x, y} для максимальной честности замера Canvas
                normalized: true,
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { boxWidth: 12, usePointStyle: true, color: '#e2e8f0', font: { size: 11 } }
                    },
                    decimation: { enabled: false }, // Отключаем клиентскую децимацию для честного рендера всех Raw-точек
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        borderColor: 'rgba(99, 102, 241, 0.4)',
                        borderWidth: 1,
                        titleColor: '#e2e8f0',
                        bodyColor: '#cbd5e1',
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(30, 41, 59, 0.6)' },
                        title: { display: true, text: 'Время t (усл. ед.)', color: '#64748b', font: { size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(30, 41, 59, 0.6)' },
                        title: { display: true, text: 'Амплитуда сигнала y(t)', color: '#64748b', font: { size: 11 } }
                    }
                }
            }
        });

        const scaleCtx = document.getElementById('scalabilityChart').getContext('2d');
        const scalabilityChart = new Chart(scaleCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Client-Side Raw T(N)',
                        data: [],
                        borderColor: '#f43f5e',
                        backgroundColor: 'rgba(244, 63, 94, 0.15)',
                        borderWidth: 2,
                        tension: 0.25,
                        fill: true
                    },
                    {
                        label: 'Server-Side LTTB T(N)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        borderWidth: 2,
                        tension: 0.25,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 250 },
                plugins: {
                    legend: { labels: { boxWidth: 10, font: { size: 10 } } }
                },
                scales: {
                    x: { grid: { color: 'rgba(30, 41, 59, 0.5)' } },
                    y: {
                        grid: { color: 'rgba(30, 41, 59, 0.5)' },
                        title: { display: true, text: 'Total Latency (мс)', font: { size: 10 } }
                    }
                }
            }
        });

        const breakdownCtx = document.getElementById('breakdownChart').getContext('2d');
        const breakdownChart = new Chart(breakdownCtx, {
            type: 'bar',
            data: {
                labels: ['Raw Pipeline', 'LTTB Pipeline'],
                datasets: [
                    { label: 'Server Compute (мс)', data: [0, 0], backgroundColor: '#6366f1' },
                    { label: 'Network + JSON Parse (мс)', data: [0, 0], backgroundColor: '#f59e0b' },
                    { label: 'Canvas Render (мс)', data: [0, 0], backgroundColor: '#f43f5e' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 250 },
                plugins: {
                    legend: { labels: { boxWidth: 10, font: { size: 10 } } }
                },
                scales: {
                    x: { stacked: true, grid: { color: 'rgba(30, 41, 59, 0.5)' } },
                    y: { stacked: true, grid: { color: 'rgba(30, 41, 59, 0.5)' }, title: { display: true, text: 'мс', font: { size: 10 } } }
                }
            }
        });

        // ----------------------------------------------------------------------
        // 3. ОБРАБОТЧИКИ ПАРАМЕТРОВ UI
        // ----------------------------------------------------------------------
        function selectN(n) {
            state.selectedN = n;
            document.getElementById('selectedNBadge').textContent = `N = ${n.toLocaleString('ru-RU')}`;
            document.querySelectorAll('.n-btn').forEach(btn => {
                const btnN = parseInt(btn.getAttribute('data-n'), 10);
                if (btnN === n) {
                    btn.className = 'n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-indigo-500 bg-indigo-500/20 text-white font-semibold transition';
                } else {
                    btn.className = 'n-btn px-2.5 py-2 rounded-lg text-xs font-mono border border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 transition';
                }
            });
        }

        function updateThreshold(val) {
            state.thresholdK = parseInt(val, 10);
            document.getElementById('thresholdLabel').textContent = `${state.thresholdK} точек`;
        }

        function updateSignalFormula() {
            const sel = document.getElementById('signalSelect').value;
            state.signalType = sel;
            const box = document.getElementById('formulaBox');
            if (sel === 'sine_noise') {
                box.textContent = 'x(t) = Σ Aᵢ·sin(ωᵢt) + N(0, σ²) + δ(t - tₖ)';
            } else if (sel === 'random_walk') {
                box.textContent = 'X(t) = Σ εᵢ / √N + A·sin(ω₀t),  εᵢ ~ N(0, 1)';
            } else {
                box.textContent = 'x(t) = Σ Aₖ·exp(-γ(t - tₖ))·cos(ω(t - tₖ))';
            }
        }

        function togglePointMarkers() {
            state.showPointMarkers = document.getElementById('togglePoints').checked;
            mainChart.data.datasets.forEach(ds => {
                if (ds.isLttb) {
                    ds.pointRadius = state.showPointMarkers ? 2.2 : 0;
                }
            });
            mainChart.update('none');
        }

        function setStatus(text, isBusy = false) {
            const badge = document.getElementById('statusBadge');
            const label = document.getElementById('statusText');
            label.textContent = text;
            if (isBusy) {
                badge.className = 'hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-mono';
            } else {
                badge.className = 'hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-mono';
            }
        }

        function formatPayload(bytes) {
            const kb = bytes / 1024;
            if (kb >= 1024) {
                return `${(kb / 1024).toFixed(2)} МБ`;
            }
            return `${kb.toFixed(1)} КБ`;
        }

        // ----------------------------------------------------------------------
        // 4. ЯДРО ИЗМЕРЕНИЯ И ПРОФИЛИРОВАНИЯ ЗАПРОСОВ
        // ----------------------------------------------------------------------
        async function fetchAndProfile(mode, n, k, signalType) {
            maxFrameGapMs = 16.7;
            trackingLagWindow = true;

            const url = mode === 'raw'
                ? `/api/generate/raw?count=${n}&signal_type=${signalType}`
                : `/api/generate/downsampled?count=${n}&threshold=${k}&signal_type=${signalType}`;

            // Фаза 1: Сетевой запрос и чтение сырого текста
            const tNetStart = performance.now();
            const response = await fetch(url);
            const rawText = await response.text();
            const tNetEnd = performance.now();

            // Вычисляем точный размер полезного груза в байтах
            const payloadBytes = new Blob([rawText]).size;

            // Фаза 2: Парсинг JSON в памяти браузера
            const tParseStart = performance.now();
            const payload = JSON.parse(rawText);
            const tParseEnd = performance.now();

            const serverMs = payload.meta.server_total_ms;
            const netTransferMs = Math.max(0.1, (tNetEnd - tNetStart) - serverMs);
            const parseMs = tParseEnd - tParseStart;

            return {
                mode,
                n,
                k: payload.meta.returned_count,
                meta: payload.meta,
                data: payload.data,
                payloadBytes,
                serverMs,
                netTransferMs,
                parseMs,
                netPlusParseMs: netTransferMs + parseMs
            };
        }

        function renderOnCanvasAndMeasure(datasets) {
            return new Promise(resolve => {
                const tRenderStart = performance.now();
                mainChart.data.datasets = datasets;
                mainChart.update('none');

                // Двойной requestAnimationFrame гарантирует завершение фактической отрисовки кадра браузером
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        const tRenderEnd = performance.now();
                        trackingLagWindow = false;
                        resolve(tRenderEnd - tRenderStart);
                    });
                });
            });
        }

        // ----------------------------------------------------------------------
        // 5. ОДИНОЧНЫЙ ЗАПУСК (RAW ИЛИ LTTB) И НАЛОЖЕНИЕ (OVERLAY)
        // ----------------------------------------------------------------------
        async function runSingleExperiment(mode) {
            const n = state.selectedN;
            const k = state.thresholdK;
            const sig = state.signalType;

            setStatus(`Выполнение ${mode.toUpperCase()} (N=${n.toLocaleString('ru-RU')})...`, true);

            const profile = await fetchAndProfile(mode, n, k, sig);

            const datasetConfig = mode === 'raw'
                ? [{
                    label: `Client-Side Raw (${n.toLocaleString('ru-RU')} точек)`,
                    data: profile.data,
                    borderColor: '#f43f5e',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    borderWidth: 1.1,
                    pointRadius: 0,
                    isLttb: false
                }]
                : [{
                    label: `Server-Side LTTB (${profile.k} точек из ${n.toLocaleString('ru-RU')})`,
                    data: profile.data,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 1.6,
                    pointRadius: state.showPointMarkers ? 2.2 : 0,
                    pointBackgroundColor: '#10b981',
                    isLttb: true
                }];

            const renderMs = await renderOnCanvasAndMeasure(datasetConfig);
            profile.renderMs = renderMs;
            profile.totalLatencyMs = profile.serverMs + profile.netPlusParseMs + renderMs;

            if (mode === 'raw') {
                state.lastRawRun[n] = profile;
            } else {
                state.lastLttbRun[n] = profile;
            }

            updateHudCards(profile);
            syncRecordAndCharts(n);

            document.getElementById('chartModeTag').textContent = mode === 'raw'
                ? `Режим: Client-Side Raw`
                : `Режим: Server-Side LTTB (K=${profile.k})`;
            document.getElementById('renderedCountInfo').textContent =
                `Отрисовано: ${profile.k.toLocaleString('ru-RU')} сегментов`;

            setStatus(`Готово (${profile.totalLatencyMs.toFixed(1)} мс)`, false);
        }

        async function runOverlayComparison() {
            const n = state.selectedN;
            const k = state.thresholdK;
            const sig = state.signalType;

            setStatus(`Верификация наложения Raw + LTTB (N=${n.toLocaleString('ru-RU')})...`, true);

            // Выполняем оба запроса последовательно для честного измерения каждого
            const rawProf = await fetchAndProfile('raw', n, k, sig);
            const rRaw = await renderOnCanvasAndMeasure([{
                label: `Raw (${n.toLocaleString('ru-RU')})`,
                data: rawProf.data,
                borderColor: 'rgba(99, 102, 241, 0.45)',
                borderWidth: 1.0,
                pointRadius: 0,
                isLttb: false
            }]);
            rawProf.renderMs = rRaw;
            rawProf.totalLatencyMs = rawProf.serverMs + rawProf.netPlusParseMs + rRaw;
            state.lastRawRun[n] = rawProf;

            const lttbProf = await fetchAndProfile('lttb', n, k, sig);
            const rLttb = await renderOnCanvasAndMeasure([{
                label: `LTTB (${lttbProf.k})`,
                data: lttbProf.data,
                borderColor: '#10b981',
                borderWidth: 1.6,
                pointRadius: 0,
                isLttb: true
            }]);
            lttbProf.renderMs = rLttb;
            lttbProf.totalLatencyMs = lttbProf.serverMs + lttbProf.netPlusParseMs + rLttb;
            state.lastLttbRun[n] = lttbProf;

            // Отрисовываем оба ряда поверх друг друга
            await renderOnCanvasAndMeasure([
                {
                    label: `Аппроксимация LTTB (${lttbProf.k} точек, R²=${lttbProf.meta.r2})`,
                    data: lttbProf.data,
                    borderColor: '#10b981',
                    borderWidth: 1.8,
                    pointRadius: state.showPointMarkers ? 2.4 : 0,
                    pointBackgroundColor: '#10b981',
                    isLttb: true,
                    order: 1
                },
                {
                    label: `Исходный ряд Raw (${n.toLocaleString('ru-RU')} точек)`,
                    data: rawProf.data,
                    borderColor: 'rgba(244, 63, 94, 0.42)',
                    borderWidth: 1.1,
                    pointRadius: 0,
                    isLttb: false,
                    order: 2
                }
            ]);

            updateHudCards(lttbProf, rawProf);
            syncRecordAndCharts(n);

            document.getElementById('chartModeTag').textContent = `Режим: Наложение Raw + LTTB (Верификация пиков)`;
            document.getElementById('renderedCountInfo').textContent =
                `Сравнение: ${n.toLocaleString('ru-RU')} vs ${lttbProf.k} точек`;

            setStatus(`Верификация завершена`, false);
        }

        // ----------------------------------------------------------------------
        // 6. ОБНОВЛЕНИЕ HUD-КАРТОЧЕК, ТАБЛИЦЫ И АНАЛИТИЧЕСКИХ ГРАФИКОВ
        // ----------------------------------------------------------------------
        function updateHudCards(activeProf, refRawProf = null) {
            const n = activeProf.n;
            const pairedRaw = refRawProf || state.lastRawRun[n];
            const pairedLttb = activeProf.mode === 'lttb' ? activeProf : state.lastLttbRun[n];

            document.getElementById('hudServerTime').textContent = `${activeProf.serverMs.toFixed(2)} мс`;
            document.getElementById('hudServerSub').textContent = activeProf.mode === 'lttb'
                ? `Ген: ${activeProf.meta.server_gen_ms.toFixed(1)} | LTTB: ${activeProf.meta.server_reduce_ms.toFixed(1)} мс`
                : `Ген: ${activeProf.meta.server_gen_ms.toFixed(1)} | Сериал: ${activeProf.meta.server_serialize_ms.toFixed(1)} мс`;

            document.getElementById('hudPayloadSize').textContent = formatPayload(activeProf.payloadBytes);
            document.getElementById('hudPayloadSub').textContent =
                `Сжатие выборок: ${activeProf.meta.compression_ratio}x`;

            document.getElementById('hudNetParseTime').textContent = `${activeProf.netPlusParseMs.toFixed(2)} мс`;
            document.getElementById('hudNetParseSub').textContent =
                `Net: ${activeProf.netTransferMs.toFixed(1)} | Parse: ${activeProf.parseMs.toFixed(1)} мс`;

            document.getElementById('hudRenderTime').textContent = `${activeProf.renderMs.toFixed(2)} мс`;
            document.getElementById('hudRenderSub').textContent =
                `Точек на Canvas: ${activeProf.k.toLocaleString('ru-RU')}`;

            document.getElementById('hudTotalTime').textContent = `${activeProf.totalLatencyMs.toFixed(2)} мс`;
            document.getElementById('hudTotalSub').textContent =
                `Режим: ${activeProf.mode.toUpperCase()}`;

            if (pairedRaw && pairedLttb) {
                const speedup = pairedRaw.totalLatencyMs / Math.max(0.1, pairedLttb.totalLatencyMs);
                document.getElementById('hudSpeedup').textContent = `${speedup.toFixed(1)}x быстрее`;
                document.getElementById('hudFidelitySub').textContent =
                    `Пики: ${pairedLttb.meta.peak_retention_pct}% | R²: ${pairedLttb.meta.r2}`;

                // Обновляем график декомпозиции для текущего N
                breakdownChart.data.datasets[0].data = [pairedRaw.serverMs, pairedLttb.serverMs];
                breakdownChart.data.datasets[1].data = [pairedRaw.netPlusParseMs, pairedLttb.netPlusParseMs];
                breakdownChart.data.datasets[2].data = [pairedRaw.renderMs, pairedLttb.renderMs];
                breakdownChart.update();
            } else {
                document.getElementById('hudSpeedup').textContent =
                    activeProf.mode === 'lttb' ? `R² = ${activeProf.meta.r2}` : `Базовый замер`;
                document.getElementById('hudFidelitySub').textContent =
                    `Пики: ${activeProf.meta.peak_retention_pct}% | RMSE: ${activeProf.meta.rmse}`;
            }
        }

        function syncRecordAndCharts(n) {
            const raw = state.lastRawRun[n];
            const lttb = state.lastLttbRun[n];

            state.tableRecords[n] = { n, raw, lttb };
            renderBenchmarkTable();
            updateScalabilityChart();
        }

        function renderBenchmarkTable() {
            const tbody = document.getElementById('benchmarkTableBody');
            const sortedKeys = Object.keys(state.tableRecords).map(Number).sort((a, b) => a - b);

            if (sortedKeys.length === 0) {
                tbody.innerHTML = `<tr><td colspan="10" class="py-6 text-center text-slate-500 font-sans">Нет данных</td></tr>`;
                return;
            }

            tbody.innerHTML = sortedKeys.map(n => {
                const rec = state.tableRecords[n];
                const r = rec.raw;
                const l = rec.lttb;

                const kText = l ? l.k : state.thresholdK;
                const rPay = r ? formatPayload(r.payloadBytes) : '—';
                const lPay = l ? formatPayload(l.payloadBytes) : '—';
                const rRnd = r ? `${r.renderMs.toFixed(1)} мс` : '—';
                const lRnd = l ? `${l.renderMs.toFixed(1)} мс` : '—';
                const rTot = r ? r.totalLatencyMs.toFixed(1) : '—';
                const lTot = l ? l.totalLatencyMs.toFixed(1) : '—';

                let speedupBadge = '—';
                if (r && l) {
                    const factor = (r.totalLatencyMs / Math.max(0.1, l.totalLatencyMs)).toFixed(1);
                    speedupBadge = `<span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-semibold">${factor}x</span>`;
                }

                const fidelityText = l
                    ? `${l.meta.peak_retention_pct}% / ${l.meta.rmse}`
                    : '—';

                return `
                    <tr class="hover:bg-slate-900/60 transition">
                        <td class="py-2.5 px-3 font-semibold text-white">${n.toLocaleString('ru-RU')}</td>
                        <td class="py-2.5 px-3 text-slate-400">${kText}</td>
                        <td class="py-2.5 px-3 text-rose-300">${rPay}</td>
                        <td class="py-2.5 px-3 text-emerald-300">${lPay}</td>
                        <td class="py-2.5 px-3 text-rose-300">${rRnd}</td>
                        <td class="py-2.5 px-3 text-emerald-300">${lRnd}</td>
                        <td class="py-2.5 px-3 font-semibold text-rose-400">${rTot}</td>
                        <td class="py-2.5 px-3 font-semibold text-emerald-400">${lTot}</td>
                        <td class="py-2.5 px-3">${speedupBadge}</td>
                        <td class="py-2.5 px-3 text-slate-300">${fidelityText}</td>
                    </tr>
                `;
            }).join('');
        }

        function updateScalabilityChart() {
            const sortedKeys = Object.keys(state.tableRecords).map(Number).sort((a, b) => a - b);
            scalabilityChart.data.labels = sortedKeys.map(n => n >= 1000 ? `${n / 1000}k` : n);
            scalabilityChart.data.datasets[0].data = sortedKeys.map(n =>
                state.tableRecords[n].raw ? Number(state.tableRecords[n].raw.totalLatencyMs.toFixed(2)) : null
            );
            scalabilityChart.data.datasets[1].data = sortedKeys.map(n =>
                state.tableRecords[n].lttb ? Number(state.tableRecords[n].lttb.totalLatencyMs.toFixed(2)) : null
            );
            scalabilityChart.update();
        }

        // ----------------------------------------------------------------------
        // 7. АВТОМАТИЧЕСКИЙ ПОЛНЫЙ СТРЕСС-ТЕСТ ПО СЕТКЕ ВЫБОРОК
        // ----------------------------------------------------------------------
        async function runFullBenchmarkSuite() {
            const sizes = [1000, 5000, 20000, 50000, 100000, 250000];
            const btn = document.getElementById('btnStressTest');
            const progContainer = document.getElementById('suiteProgressContainer');
            const progBar = document.getElementById('suiteProgressBar');
            const progLabel = document.getElementById('suiteProgressLabel');
            const progPct = document.getElementById('suiteProgressPct');

            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            progContainer.classList.remove('hidden');
            progContainer.classList.add('flex');

            for (let i = 0; i < sizes.length; i++) {
                const n = sizes[i];
                selectN(n);

                const pct1 = Math.round(((i * 2) / (sizes.length * 2)) * 100);
                progBar.style.width = `${pct1}%`;
                progPct.textContent = `${pct1}%`;
                progLabel.textContent = `Тест Raw: N = ${n.toLocaleString('ru-RU')}...`;

                await runSingleExperiment('raw');
                await new Promise(r => setTimeout(r, 120));

                const pct2 = Math.round(((i * 2 + 1) / (sizes.length * 2)) * 100);
                progBar.style.width = `${pct2}%`;
                progPct.textContent = `${pct2}%`;
                progLabel.textContent = `Тест LTTB: N = ${n.toLocaleString('ru-RU')}...`;

                await runSingleExperiment('lttb');
                await new Promise(r => setTimeout(r, 120));
            }

            // В финале включаем режим наложения на 250 000 точек для эффектной демонстрации
            progBar.style.width = '100%';
            progPct.textContent = '100%';
            progLabel.textContent = 'Стресс-тест успешно завершен';
            await runOverlayComparison();

            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        }

        // ----------------------------------------------------------------------
        // 8. ЭКСПОРТ РЕЗУЛЬТАТОВ В CSV И LATEX ДЛЯ НАУЧНОЙ СТАТЬИ
        // ----------------------------------------------------------------------
        function exportTableToCSV() {
            const sortedKeys = Object.keys(state.tableRecords).map(Number).sort((a, b) => a - b);
            if (sortedKeys.length === 0) {
                alert('Сначала выполните замеры или запустите стресс-тест.');
                return;
            }

            const headers = [
                'N_Points', 'LTTB_K',
                'Raw_Payload_KB', 'LTTB_Payload_KB',
                'Raw_Server_ms', 'LTTB_Server_ms',
                'Raw_NetParse_ms', 'LTTB_NetParse_ms',
                'Raw_Render_ms', 'LTTB_Render_ms',
                'Raw_Total_ms', 'LTTB_Total_ms',
                'Speedup_Factor', 'Peak_Retention_Pct', 'RMSE', 'R2'
            ];

            const rows = sortedKeys.map(n => {
                const { raw: r, lttb: l } = state.tableRecords[n];
                const speedup = (r && l) ? (r.totalLatencyMs / Math.max(0.1, l.totalLatencyMs)).toFixed(2) : '';
                return [
                    n,
                    l ? l.k : state.thresholdK,
                    r ? (r.payloadBytes / 1024).toFixed(2) : '',
                    l ? (l.payloadBytes / 1024).toFixed(2) : '',
                    r ? r.serverMs.toFixed(2) : '',
                    l ? l.serverMs.toFixed(2) : '',
                    r ? r.netPlusParseMs.toFixed(2) : '',
                    l ? l.netPlusParseMs.toFixed(2) : '',
                    r ? r.renderMs.toFixed(2) : '',
                    l ? l.renderMs.toFixed(2) : '',
                    r ? r.totalLatencyMs.toFixed(2) : '',
                    l ? l.totalLatencyMs.toFixed(2) : '',
                    speedup,
                    l ? l.meta.peak_retention_pct : '',
                    l ? l.meta.rmse : '',
                    l ? l.meta.r2 : ''
                ].join(',');
            });

            const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement('a');
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', `nir_timeseries_benchmark_${state.signalType}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function copyLatexTable() {
            const sortedKeys = Object.keys(state.tableRecords).map(Number).sort((a, b) => a - b);
            if (sortedKeys.length === 0) {
                alert('Таблица пуста. Запустите стресс-тест перед генерацией LaTeX.');
                return;
            }

            let latex = "\\begin{table}[htbp]\n\\centering\n";
            latex += "\\caption{Сравнительный анализ производительности Client-Side Raw и Server-Side LTTB}\n";
            latex += "\\begin{tabular}{|r|r|r|r|r|r|r|}\n\\hline\n";
            latex += "$N$ & Raw (КБ) & LTTB (КБ) & $T_{\\text{raw}}$ (мс) & $T_{\\text{lttb}}$ (мс) & Ускорение & $R^2$ \\\\ \\hline\n";

            sortedKeys.forEach(n => {
                const { raw: r, lttb: l } = state.tableRecords[n];
                if (r && l) {
                    const rKb = (r.payloadBytes / 1024).toFixed(1);
                    const lKb = (l.payloadBytes / 1024).toFixed(1);
                    const speedup = (r.totalLatencyMs / Math.max(0.1, l.totalLatencyMs)).toFixed(1);
                    latex += `${n} & ${rKb} & ${lKb} & ${r.totalLatencyMs.toFixed(1)} & ${l.totalLatencyMs.toFixed(1)} & ${speedup}$\\times$ & ${l.meta.r2} \\\\ \\hline\n`;
                }
            });

            latex += "\\end{tabular}\n\\end{table}";
            navigator.clipboard.writeText(latex);
            setStatus('Код LaTeX-таблицы скопирован в буфер обмена', false);
        }

        function clearBenchmarkTable() {
            state.tableRecords = {};
            state.lastRawRun = {};
            state.lastLttbRun = {};
            renderBenchmarkTable();
            updateScalabilityChart();
        }

        // Автоматический стартовый прогон при открытии страницы (на N = 50 000)
        window.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => {
                runOverlayComparison();
            }, 250);
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_workbench():
    return HTMLResponse(content=HTML_TEMPLATE)


if __name__ == "__main__":
    print("=" * 72)
    print(" ИССЛЕДОВАТЕЛЬСКИЙ СТЕНД НИР ЗАПУЩЕН")
    print(" Откройте в браузере: http://127.0.0.1:8000")
    print("=" * 72)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")