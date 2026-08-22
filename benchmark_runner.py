import sys
import time
import json
import asyncio
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.dataset_loader import MSMARCOIndicLoader
from backend.vector_engine import HighPerformanceVectorEngine
from backend.guardrails_engine import GuardrailsEngine
from backend.model_harness import ModelHarness
from backend.latency_profiler import LatencyProfiler
from backend.app import TextQueryRequest, process_text_query

async def run_benchmark_queries(test_queries):
    runs = []
    for item in test_queries:
        t0 = time.perf_counter()
        result = await process_text_query(TextQueryRequest(
            query=item['query'],
            language=item['language'],
            strategy='recursive_semantic'
        ))
        runs.append({
            **result.get('latency_breakdown', {}),
            'total_e2e_ms': (time.perf_counter() - t0) * 1000
        })
    return runs

def run_standalone_benchmark():
    print("=" * 65)
    print("   HH GOA 2026: v-askbot LATENCY & RAG BENCHMARK SUITE")
    print("=" * 65)

    loader = MSMARCOIndicLoader()
    vector_engine = HighPerformanceVectorEngine()
    vector_engine.index_dataset(loader)

    guardrails = GuardrailsEngine()
    harness = ModelHarness()
    profiler = LatencyProfiler(target_ms=200.0)

    # Use every available language sample so percentile results are not based
    # on a hand-picked subset.
    test_queries = []
    for language in ('en', 'hi', 'bn', 'te', 'ta', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'):
        test_queries.extend(loader.get_sample_queries(language))

    print(f"Total benchmark test queries: {len(test_queries)}")
    print("Executing the actual translation → RAG → groundedness pipeline...")

    runs = asyncio.run(run_benchmark_queries(test_queries))
    for metrics in runs:
        profiler.record_run(metrics)

    pcts = profiler.get_percentiles()
    
    print("-" * 65)
    print("📊 BENCHMARK RESULTS SUMMARY:")
    print(f"  • Total Runs Evaluated:  {pcts['sample_count']}")
    print(f"  • Target E2E Latency:    < {pcts['target_ms']} ms")
    print(f"  • Sub-200ms Pass Rate:   {pcts['under_target_rate']}")
    print("-" * 65)
    print("⚡ PERCENTILE LATENCIES:")
    print(f"  • Total E2E  -> P50: {pcts['total_e2e']['p50']:>6.1f} ms | P70: {pcts['total_e2e']['p70']:>6.1f} ms | P100: {pcts['total_e2e']['p100']:>6.1f} ms")
    print(f"  • Retrieval  -> P50: {pcts['retrieval']['p50']:>6.1f} ms | P70: {pcts['retrieval']['p70']:>6.1f} ms | P100: {pcts['retrieval']['p100']:>6.1f} ms")
    print(f"  • LLM (TTFT) -> P50: {pcts['llm_ttft']['p50']:>6.1f} ms | P70: {pcts['llm_ttft']['p70']:>6.1f} ms | P100: {pcts['llm_ttft']['p100']:>6.1f} ms")
    print(f"  • Guardrails -> P50: {pcts['guardrails']['p50']:>6.1f} ms | P70: {pcts['guardrails']['p70']:>6.1f} ms | P100: {pcts['guardrails']['p100']:>6.1f} ms")
    print("=" * 65)
    if pcts['under_target_rate'] == '100.0%':
        print("PASS: measured runs stayed under the target.")
    else:
        print("FAIL: measured end-to-end latency does not meet the target.")

if __name__ == '__main__':
    run_standalone_benchmark()
