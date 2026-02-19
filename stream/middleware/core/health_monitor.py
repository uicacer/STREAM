"""
Background health monitor for AI service tiers.

DISABLED: This module previously ran health checks every 5 minutes in a
background thread. This was removed because:

  1. At scale (thousands of users), periodic health checks overwhelm
     Lakeshore — each check submits a real 1-token inference job through
     Globus Compute, consuming GPU time and Globus API quota.

  2. Health checks are now ON-DEMAND only — triggered by user actions
     (selecting a tier, changing a model) instead of a timer.

  3. The frontend no longer polls every 30 seconds either. Combined with
     this change, STREAM only checks health when the user actually needs
     to know, keeping Lakeshore free for real inference work.

The HealthMonitor class is kept as a no-op stub so existing imports
(lifecycle.py) don't break. start() and stop() do nothing.
"""

import logging

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Stub — background health polling has been removed (on-demand only now)."""

    def __init__(self, check_interval: int = 300):
        pass

    def start(self):
        """No-op. Health checks are now on-demand only."""
        logger.info("Health monitor disabled (on-demand health checks only)")

    def stop(self):
        """No-op."""
        pass


# Singleton instance (kept for import compatibility)
health_monitor = HealthMonitor()
