"""
Performance metrics tracking for STREAM.

This module tracks:
- Time To First Token (TTFT)
- End-to-end latency
- Tokens per second (throughput)
- Request success/failure rates
- Tier usage distribution
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# dataclass is usually used when you want to group related data together
# In this case, we are grouping all metrics related to a single request
@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    correlation_id: str
    tier: str
    model: str
    complexity: str

    # Timing metrics (milliseconds)
    ttft_ms: float | None = None
    total_latency_ms: float | None = None

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float | None = None

    # Cost metrics
    total_cost: float = 0.0

    # Status
    success: bool = True
    error_message: str | None = None
    fallback_used: bool = False
    tiers_tried: list = None


class MetricsTracker:
    """Tracks and logs performance metrics."""

    def __init__(self):
        self.start_time = None
        self.first_token_time = None
        self.metrics = None

    def start_request(self, correlation_id: str, tier: str, model: str, complexity: str):
        """Start tracking a request."""
        self.start_time = time.perf_counter()
        self.metrics = RequestMetrics(
            correlation_id=correlation_id,
            tier=tier,
            model=model,
            complexity=complexity,
            tiers_tried=[tier],
        )

    def record_first_token(self):
        """Record time to first token."""
        if self.first_token_time is None and self.start_time:
            self.first_token_time = time.perf_counter()
            self.metrics.ttft_ms = (self.first_token_time - self.start_time) * 1000

            logger.info(
                f"[{self.metrics.correlation_id}] TTFT: {self.metrics.ttft_ms:.2f}ms",
                extra={
                    "correlation_id": self.metrics.correlation_id,
                    "ttft_ms": self.metrics.ttft_ms,
                    "tier": self.metrics.tier,
                },
            )

    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token counts."""
        self.metrics.input_tokens = input_tokens
        self.metrics.output_tokens = output_tokens

    def record_completion(self, cost: float = 0.0):
        """Record successful completion."""
        if self.start_time:
            end_time = time.perf_counter()
            self.metrics.total_latency_ms = (end_time - self.start_time) * 1000

            if self.metrics.output_tokens > 0:
                duration_seconds = end_time - (self.first_token_time or self.start_time)
                self.metrics.tokens_per_second = self.metrics.output_tokens / duration_seconds

            self.metrics.total_cost = cost
            self.metrics.success = True

            self._log_metrics()

    def record_error(self, error_message: str):
        """Record failed request."""
        if self.start_time:
            end_time = time.perf_counter()
            self.metrics.total_latency_ms = (end_time - self.start_time) * 1000
            self.metrics.success = False
            self.metrics.error_message = error_message

            self._log_metrics()

    def record_fallback(self, new_tier: str):
        """Record tier fallback."""
        self.metrics.fallback_used = True
        self.metrics.tiers_tried.append(new_tier)
        self.metrics.tier = new_tier

    def _log_metrics(self):
        """Log complete metrics."""
        logger.info(
            f"[{self.metrics.correlation_id}] Request complete",
            extra={
                "correlation_id": self.metrics.correlation_id,
                "tier": self.metrics.tier,
                "model": self.metrics.model,
                "complexity": self.metrics.complexity,
                "ttft_ms": self.metrics.ttft_ms,
                "total_latency_ms": self.metrics.total_latency_ms,
                "input_tokens": self.metrics.input_tokens,
                "output_tokens": self.metrics.output_tokens,
                "tokens_per_second": self.metrics.tokens_per_second,
                "total_cost": self.metrics.total_cost,
                "success": self.metrics.success,
                "fallback_used": self.metrics.fallback_used,
                "tiers_tried": self.metrics.tiers_tried,
                "error": self.metrics.error_message,
            },
        )
