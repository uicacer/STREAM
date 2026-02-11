"""
Background health monitor for AI service tiers.

This module runs health checks in a background thread to keep the
tier health cache fresh, preventing blocking during user requests.
"""

import logging
import threading

from stream.middleware.core.tier_health import _tier_health, update_tier_health

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Background thread that keeps tier health cache fresh."""

    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self._thread = None
        self._stop_event = threading.Event()
        # Track previous status to only log changes
        self._previous_status = {}

    def start(self):
        """Start the background health check thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_checks, daemon=True)
        self._thread.start()
        logger.info(f"Health monitor started (interval: {self.check_interval}s)")

    def stop(self):
        """Stop the background health check thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Health monitor stopped")

    def _run_checks(self):
        """Main loop: check all tiers periodically."""
        tiers = ["local", "lakeshore", "cloud"]

        while not self._stop_event.is_set():
            for tier in tiers:
                if self._stop_event.is_set():
                    break
                try:
                    # Update health (includes retries in check_tier_health)
                    update_tier_health(tier)

                    # Only log if status changed
                    current_status = _tier_health[tier]["available"]
                    previous = self._previous_status.get(tier)

                    if previous is not None and current_status != previous:
                        if current_status:
                            logger.info(f"Tier {tier.upper()} is now AVAILABLE")
                        else:
                            logger.warning(f"Tier {tier.upper()} is now UNAVAILABLE")

                    self._previous_status[tier] = current_status

                except Exception as e:
                    logger.error(f"Health check failed for {tier}: {e}")

            # Wait for next check interval (interruptible)
            self._stop_event.wait(self.check_interval)


# Singleton instance
health_monitor = HealthMonitor(check_interval=300)  # 5 minutes
