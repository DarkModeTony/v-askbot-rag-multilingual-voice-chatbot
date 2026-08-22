import time
from typing import List, Dict, Any
import numpy as np

class LatencyProfiler:
    def __init__(self, target_ms: float = 200.0):
        self.target_ms = target_ms
        self.history: List[Dict[str, float]] = []

    def record_run(self, metrics: Dict[str, float]) -> None:
        self.history.append(metrics)
        if len(self.history) > 500:
            self.history.pop(0)

    def get_percentiles(self) -> Dict[str, Any]:
        if not self.history:
            return {
                'total_e2e': {'p50': None, 'p70': None, 'p100': None},
                'retrieval': {'p50': None, 'p70': None, 'p100': None},
                'llm_ttft': {'p50': None, 'p70': None, 'p100': None},
                'guardrails': {'p50': None, 'p70': None, 'p100': None},
                'sample_count': 0,
                'target_ms': self.target_ms,
                'under_target_rate': 'N/A'
            }

        e2e_latencies = [m.get('total_e2e_ms', 0.0) for m in self.history]
        retrieval_latencies = [m.get('retrieval_ms', 0.0) for m in self.history]
        llm_latencies = [m.get('llm_ttft_ms', 0.0) for m in self.history]
        guardrail_latencies = [m.get('guardrails_ms', 0.0) for m in self.history]

        under_target = sum(1 for x in e2e_latencies if x <= self.target_ms)
        under_target_pct = (under_target / len(e2e_latencies)) * 100

        return {
            'total_e2e': {
                'p50': round(float(np.percentile(e2e_latencies, 50)), 1),
                'p70': round(float(np.percentile(e2e_latencies, 70)), 1),
                'p100': round(float(np.percentile(e2e_latencies, 100)), 1)
            },
            'retrieval': {
                'p50': round(float(np.percentile(retrieval_latencies, 50)), 1),
                'p70': round(float(np.percentile(retrieval_latencies, 70)), 1),
                'p100': round(float(np.percentile(retrieval_latencies, 100)), 1)
            },
            'llm_ttft': {
                'p50': round(float(np.percentile(llm_latencies, 50)), 1),
                'p70': round(float(np.percentile(llm_latencies, 70)), 1),
                'p100': round(float(np.percentile(llm_latencies, 100)), 1)
            },
            'guardrails': {
                'p50': round(float(np.percentile(guardrail_latencies, 50)), 1),
                'p70': round(float(np.percentile(guardrail_latencies, 70)), 1),
                'p100': round(float(np.percentile(guardrail_latencies, 100)), 1)
            },
            'sample_count': len(self.history),
            'target_ms': self.target_ms,
            'under_target_rate': f"{under_target_pct:.1f}%"
        }
