import os
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

# Suppress OpenMP library conflicts and segfaults on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from mcp_vector_shield.mcp_classifier_engine import MCPNeuralShield

class ComprehensiveBenchmark:
    def __init__(self):
        print("=" * 95)
        print(
            "         MCPSecurity: Multi-Dataset Comprehensive Neural Shield Classifier Benchmark        "
        )
        print("=" * 95)

        # 1. Initialize MCPNeuralShield
        print("[1/4] Initializing MCPNeuralShield (model: 'shield_model.pt')...")
        self.registry = MCPNeuralShield(model_path="shield_model.pt", device="cpu")
        print(f"      Neural Shield active on device: {self.registry.device}")

        # Thread pool executor for async CPU-bound inference scheduling
        cores = os.cpu_count() or 4
        self.executor = ThreadPoolExecutor(max_workers=min(32, cores * 4))

    def load_datasets(
        self,
    ) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
    ]:
        """
        Loads all three localized security datasets.
        """
        print("\n[2/4] Ingesting localized benchmark JSON datasets...")

        # A. MCPTox Dataset
        with open("safe_baselines.json", "r") as f:
            mcptox_baselines = json.load(f)
        with open("poisoned_tests.json", "r") as f:
            mcptox_poisoned = json.load(f)
        print(
            f"      Ingested MCPTox Dataset: {len(mcptox_baselines)} Baselines, {len(mcptox_poisoned)} Poisoned Test Cases."
        )

        # B. MCPSecBench Dataset
        with open("secbench_shadow_tests.json", "r") as f:
            secbench_shadow = json.load(f)
        print(
            f"      Ingested MCPSecBench Dataset: {len(secbench_shadow)} Shadow/Exploit Test Cases."
        )

        # C. MCPToolBench++ Dataset
        with open("massive_safe_baselines.json", "r") as f:
            toolbench_baselines = json.load(f)
        print(
            f"      Ingested MCPToolBench++ Dataset: {len(toolbench_baselines)} Enterprise Legitimate Tools."
        )

        return mcptox_baselines, mcptox_poisoned, secbench_shadow, toolbench_baselines

    async def _async_is_attack(self, tool: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Asynchronously runs is_attack on the thread pool to avoid blocking the asyncio loop.
        """
        loop = asyncio.get_running_loop()
        start = time.time()
        # Schedule the CPU-bound Neural inference lookup on the thread pool
        is_atk = await loop.run_in_executor(
            self.executor, self.registry.is_attack, tool
        )
        latency = (time.time() - start) * 1000  # Convert to ms
        return is_atk, latency

    async def evaluate_suite_async(
        self, name: str, tools: List[Dict[str, Any]], expected_attack: bool
    ) -> Dict[str, Any]:
        """
        Runs async batch processing over an entire dataset split and computes metrics.
        """
        print(f"      Batch evaluating: {name} ({len(tools)} items)...")
        tasks = [self._async_is_attack(tool) for tool in tools]
        results = await asyncio.gather(*tasks)

        flagged_count = sum(1 for is_atk, _ in results if is_atk)
        latencies = [latency for _, latency in results]

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate metric rates
        if expected_attack:
            tpr = (flagged_count / len(tools)) * 100
            fpr = 0.0
        else:
            tpr = 100.0
            fpr = (flagged_count / len(tools)) * 100

        return {
            "name": name,
            "count": len(tools),
            "flagged": flagged_count,
            "avg_latency": avg_latency,
            "tpr": tpr,
            "fpr": fpr,
        }


async def main():
    benchmark = ComprehensiveBenchmark()

    # 2. Load localized data
    mcptox_baselines, mcptox_poisoned, secbench_shadow, toolbench_baselines = (
        benchmark.load_datasets()
    )

    # 3. Sequentially execute comprehensive asynchronous evaluation splits
    print("\n[3/4] Running asynchronous multi-dataset evaluation suites...")

    # Suite A: MCPTox poisoned shadowing attacks (Expected: Attack)
    tox_poisoned_result = await benchmark.evaluate_suite_async(
        "MCPTox (Prompt Injections / Poisoning)", mcptox_poisoned, expected_attack=True
    )

    # Suite B: MCPSecBench Shadow Server & Malicious Client exploits (Expected: Attack)
    secbench_result = await benchmark.evaluate_suite_async(
        "MCPSecBench (Shadow Server & CVE Exploits)", secbench_shadow, expected_attack=True
    )

    # Suite C: MCPToolBench++ Identical baselines (Expected: Benign)
    toolbench_identical_result = await benchmark.evaluate_suite_async(
        "MCPToolBench++ (Legitimate Identical Queries)", toolbench_baselines, expected_attack=False
    )

    # Suite D: MCPToolBench++ Harmless minor updates (Expected: Benign)
    toolbench_updates = []
    for tool in toolbench_baselines:
        toolbench_updates.append(
            {
                "name": tool["name"],
                "description": tool["description"] + " Optimized for production deployment.",
                "inputSchema": tool["inputSchema"],
            }
        )
    toolbench_updates_result = await benchmark.evaluate_suite_async(
        "MCPToolBench++ (Legitimate Harmless Updates)", toolbench_updates, expected_attack=False
    )

    # 4. Output Console LaTeX/Markdown Structured Report
    print("\n[4/4] Compiling structured final benchmark summary report...")
    print("\n" + "=" * 105)
    print(
        "                                      FINAL SECURITY BENCHMARK REPORT                                    "
    )
    print("=" * 105)
    print(
        f"| {'Dataset Security Evaluation Split':<45} | {'Size':<6} | {'Flagged':<7} | {'Avg Latency':<12} | {'TPR':<7} | {'FPR':<7} |"
    )
    print("-" * 105)

    for res in [
        tox_poisoned_result,
        secbench_result,
        toolbench_identical_result,
        toolbench_updates_result,
    ]:
        print(
            f"| {res['name']:<45} | "
            f"{res['count']:<6} | "
            f"{res['flagged']:<7} | "
            f"{res['avg_latency']:8.3f} ms | "
            f"{res['tpr']:5.2f}% | "
            f"{res['fpr']:5.2f}% |"
        )

    print("-" * 105)

    # Compute overall summary metrics
    total_poisoned = tox_poisoned_result["count"] + secbench_result["count"]
    flagged_poisoned = tox_poisoned_result["flagged"] + secbench_result["flagged"]
    overall_tpr = (flagged_poisoned / total_poisoned) * 100

    total_benign = toolbench_identical_result["count"] + toolbench_updates_result["count"]
    flagged_benign = toolbench_identical_result["flagged"] + toolbench_updates_result["flagged"]
    overall_fpr = (flagged_benign / total_benign) * 100

    overall_accuracy = (
        (flagged_poisoned + (total_benign - flagged_benign)) / (total_poisoned + total_benign)
    ) * 100

    print(
        f"| {'OVERALL ATTACK DETECTION RATE (True Positive Rate)':<45} | {'':<6} | {'':<7} | {'':<12} | {overall_tpr:5.2f}% | {'':<7} |"
    )
    print(
        f"| {'OVERALL FALSE ALARM RATE (False Positive Rate)':<45} | {'':<6} | {'':<7} | {'':<12} | {'':<7} | {overall_fpr:5.2f}% |"
    )
    print(
        f"| {'OVERALL CLASSIFICATION ACCURACY':<45} | {'':<6} | {'':<7} | {'':<12} | {overall_accuracy:5.2f}% | {'':<7} |"
    )
    print("=" * 105)


if __name__ == "__main__":
    asyncio.run(main())

