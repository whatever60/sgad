from __future__ import annotations

import csv
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgad.pairwise import needleman_wunsch as py_nw
from sgad.pairwise_3d import needleman_wunsch_3d as py_nw3
from sgad.rust.pairwise import make_rust_score_scaler
from sgad.rust.pairwise import needleman_wunsch as rs_nw
from sgad.rust.pairwise_3d import needleman_wunsch_3d as rs_nw3
from sgad_rust_native import __file__ as _sgad_rust_native_file

MAT = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _median_runtime(fn, repeats: int = 3) -> float:
    values: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return float(np.median(values))


def _fit_exponent(ns: list[int], ts: list[float]) -> float:
    x = np.log(np.array(ns, dtype=float))
    y = np.log(np.array(ts, dtype=float))
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def _pairwise_dp_bytes(n: int) -> int:
    # DP + pointer tensors for 3 states in pairwise implementation.
    cells = (n + 1) * (n + 1)
    float_bytes = 8
    int8_bytes = 1
    return 3 * cells * (float_bytes + int8_bytes + int8_bytes + int8_bytes)


def _pairwise_3d_dp_bytes(n: int) -> int:
    # DP + 4 pointer tensors for 7 states in 3D implementation.
    cells = (n + 1) * (n + 1) * (n + 1)
    float_bytes = 8
    int8_bytes = 1
    return 7 * cells * (float_bytes + int8_bytes + int8_bytes + int8_bytes + int8_bytes)


def _run_once(queue: mp.Queue, fn, args: tuple, kwargs: dict) -> None:
    try:
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        queue.put(("ok", elapsed))
    except Exception as exc:  # pragma: no cover - worker path
        queue.put(("err", repr(exc)))


def _timed_run(fn, args: tuple, kwargs: dict, timeout_s: float) -> tuple[str, float | str]:
    queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(target=_run_once, args=(queue, fn, args, kwargs))
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ("timeout", timeout_s)

    if queue.empty():
        return ("err", "no result from worker")

    status, payload = queue.get_nowait()
    return (status, payload)


def _write_series_csv(
    series: dict[tuple[str, str], dict[str, list[float] | list[int]]],
    csv_path: Path,
) -> None:
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["dimension", "backend", "n", "time_s"])
        for dim in ("2d", "3d"):
            for backend in ("python", "rust"):
                ns = series[(dim, backend)]["n"]
                ts = series[(dim, backend)]["t"]
                for n, t in zip(ns, ts):
                    writer.writerow([dim, backend, n, t])


def _read_series_csv(
    csv_path: Path,
) -> dict[tuple[str, str], dict[str, list[float] | list[int]]]:
    series: dict[tuple[str, str], dict[str, list[float] | list[int]]] = {
        ("2d", "python"): {"n": [], "t": []},
        ("2d", "rust"): {"n": [], "t": []},
        ("3d", "python"): {"n": [], "t": []},
        ("3d", "rust"): {"n": [], "t": []},
    }

    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            dim = row["dimension"]
            backend = row["backend"]
            key = (dim, backend)
            if key not in series:
                continue
            series[key]["n"].append(int(row["n"]))
            series[key]["t"].append(float(row["time_s"]))

    return series


def _plot_from_csv(csv_path: Path, fig_path: Path) -> dict[tuple[str, str], dict[str, list[float] | list[int]]]:
    series = _read_series_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    color_map = {"2d": "tab:blue", "3d": "tab:orange"}
    style_map = {"python": "--", "rust": "-"}

    for idx, dim in enumerate(("2d", "3d")):
        ax = axes[idx]
        for backend in ("python", "rust"):
            ns = series[(dim, backend)]["n"]
            ts = series[(dim, backend)]["t"]
            if len(ns) == 0:
                continue

            label = f"{dim} / {backend}"
            if len(ns) >= 2:
                exp = _fit_exponent(ns, ts)
                label = f"{label} (slope~{exp:.2f})"
            ax.plot(
                ns,
                ts,
                linestyle=style_map[backend],
                color=color_map[dim],
                marker="o",
                label=label,
            )

            ax.set_title(f"Needleman-Wunsch {dim.upper()}")
            ax.set_xlabel("Sequence length n")
            ax.set_ylabel("Runtime (s)")
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.legend()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    return series


def _print_native_build_hint() -> None:
    native_path = Path(_sgad_rust_native_file)
    print(f"Native module entry: {native_path}")
    print(
        "Hint: if Rust timings look unexpectedly slow, rebuild with: "
        "`uv run maturin develop --release -m rust/Cargo.toml`."
    )


def _print_common_speedup(series: dict[tuple[str, str], dict[str, list[float] | list[int]]], dim: str) -> None:
    py_ns = list(series[(dim, "python")]["n"])
    py_ts = list(series[(dim, "python")]["t"])
    rs_ns = list(series[(dim, "rust")]["n"])
    rs_ts = list(series[(dim, "rust")]["t"])

    py_map = {int(n): float(t) for n, t in zip(py_ns, py_ts)}
    rs_map = {int(n): float(t) for n, t in zip(rs_ns, rs_ts)}
    common_ns = sorted(set(py_map) & set(rs_map))

    if not common_ns:
        print(f"{dim.upper()} speedup (common n): no overlap")
        return

    ratios = [py_map[n] / rs_map[n] for n in common_ns]
    tail_n = common_ns[-1]
    tail_speedup = ratios[-1]
    median_speedup = float(np.median(np.array(ratios, dtype=float)))
    print(
        f"{dim.upper()} speedup (common n): "
        f"tail n={tail_n} => {tail_speedup:.2f}x, median={median_speedup:.2f}x"
    )


def main() -> None:
    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "time_complexity.csv"
    fig_path = out_dir / "time_complexity.png"

    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for plot-only mode: {csv_path}")
        _print_native_build_hint()
        _plot_from_csv(csv_path, fig_path)
        print(f"Wrote {fig_path}")
        return

    rng = random.Random(123)

    timeout_s = 60.0

    # 2D request: 500..10000 step 500. 3D uses a smaller linear sweep.
    two_d_sizes = [20] + list(range(500, 10001, 500))
    three_d_sizes = list(range(20, 1001, 20))

    max_pairwise_bytes = 1_500_000_000
    max_3d_bytes = 1_500_000_000

    series: dict[tuple[str, str], dict[str, list[float] | list[int]]] = {
        ("2d", "python"): {"n": [], "t": []},
        ("2d", "rust"): {"n": [], "t": []},
        ("3d", "python"): {"n": [], "t": []},
        ("3d", "rust"): {"n": [], "t": []},
    }

    call_map = {
        ("2d", "python"): py_nw,
        ("2d", "rust"): rs_nw,
        ("3d", "python"): py_nw3,
        ("3d", "rust"): rs_nw3,
    }
    rust_2d_scaler = make_rust_score_scaler(decay_exponent=1.0, temperature=1.0)

    print("Starting benchmark scan")
    print(f"Per-alignment timeout: {timeout_s:.0f}s")
    _print_native_build_hint()

    run_plan = [
        ("2d", "rust", two_d_sizes),
        ("2d", "python", two_d_sizes),
        ("3d", "rust", three_d_sizes),
        ("3d", "python", three_d_sizes),
    ]

    for dim, backend, sizes in run_plan:
        fn = call_map[(dim, backend)]
        print(f"\n[{dim}/{backend}] scan start with {len(sizes)} candidate sizes")

        for n in sizes:
            if dim == "2d":
                need_bytes = _pairwise_dp_bytes(n)
                if need_bytes > max_pairwise_bytes:
                    print(
                        f"[{dim}/{backend}] stop at n={n}: estimated memory {need_bytes / 1e9:.2f} GB > cap"
                    )
                    break
                seqs = (_rand_seq(rng, n), _rand_seq(rng, n))
            else:
                need_bytes = _pairwise_3d_dp_bytes(n)
                if need_bytes > max_3d_bytes:
                    print(
                        f"[{dim}/{backend}] stop at n={n}: estimated memory {need_bytes / 1e9:.2f} GB > cap"
                    )
                    break
                seqs = (_rand_seq(rng, n), _rand_seq(rng, n), _rand_seq(rng, n))

            kwargs = {"score_matrix": MAT}
            if dim == "2d" and backend == "rust":
                kwargs["score_scale_fn"] = rust_2d_scaler
            print(f"[{dim}/{backend}] n={n}: running...")
            status, payload = _timed_run(fn, seqs, kwargs, timeout_s=timeout_s)

            if status == "ok":
                elapsed = float(payload)
                series[(dim, backend)]["n"].append(n)
                series[(dim, backend)]["t"].append(elapsed)
                print(f"[{dim}/{backend}] n={n}: {elapsed:.3f}s")
                continue

            if status == "timeout":
                print(
                    f"[{dim}/{backend}] stop at n={n}: exceeded {timeout_s:.0f}s timeout; skipping larger n"
                )
                break

            print(f"[{dim}/{backend}] stop at n={n}: error {payload}")
            break

    _write_series_csv(series, csv_path)
    series = _plot_from_csv(csv_path, fig_path)

    print("\nBenchmark summary")
    for dim in ("2d", "3d"):
        for backend in ("python", "rust"):
            ns = series[(dim, backend)]["n"]
            ts = series[(dim, backend)]["t"]
            if len(ns) == 0:
                print(f"- {dim}/{backend}: no completed points")
            elif len(ns) == 1:
                print(f"- {dim}/{backend}: 1 point (n={ns[0]}, t={ts[0]:.3f}s)")
            else:
                exp = _fit_exponent(ns, ts)
                print(
                    f"- {dim}/{backend}: {len(ns)} points, n_max={ns[-1]}, slope~{exp:.3f}, last={ts[-1]:.3f}s"
                )

    _print_common_speedup(series, "2d")
    _print_common_speedup(series, "3d")

    print(f"Wrote {fig_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
