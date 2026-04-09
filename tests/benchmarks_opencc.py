import gc
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencc_purepy.core import OpenCC


def _mean(values: Sequence[float]) -> float:
    return sum(values, 0.0) / len(values) if values else 0.0


def benchmark_conversion_scenarios(input_text: str, config: str = "s2t", rounds: int = 20):
    cold_total_ms = []
    post_init_cold_ms = []
    warm_ms = []

    for _ in range(rounds):
        gc.collect()

        start = time.perf_counter()
        opencc = OpenCC(config)
        _ = opencc.convert(input_text)
        end = time.perf_counter()
        cold_total_ms.append((end - start) * 1000)

        gc.collect()

        opencc = OpenCC(config)
        start = time.perf_counter()
        _ = opencc.convert(input_text)
        end = time.perf_counter()
        post_init_cold_ms.append((end - start) * 1000)

        gc.collect()

        opencc = OpenCC(config)
        _ = opencc.convert(input_text)
        start = time.perf_counter()
        _ = opencc.convert(input_text)
        end = time.perf_counter()
        warm_ms.append((end - start) * 1000)

    print(
        f"Input size: {len(input_text):>7} chars"
        f" | cold_total: {_mean(cold_total_ms):8.3f} ms"
        f" | post_init_cold: {_mean(post_init_cold_ms):8.3f} ms"
        f" | warm: {_mean(warm_ms):8.3f} ms"
        f" | rounds={rounds}"
    )


if __name__ == "__main__":
    sample = (
        "潦水尽而寒潭清，烟光凝而暮山紫。俨骖𬴂于上路，访风景于崇阿；"
        "临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。"
    )

    for size in [100, 1_000, 10_000, 100_000]:
        text = (sample * (size // len(sample) + 1))[:size]
        benchmark_conversion_scenarios(text)
