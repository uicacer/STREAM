"""
Streamlit Frontend Configuration
"""

# =============================================================================
# UI SETTINGS
# =============================================================================

APP_TITLE = "STREAM"
APP_SUBTITLE = "Smart Tiered Routing Engine for AI Models"
APP_ICON = "🌊"

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = [
    "What is Python?",
    "Explain how neural networks work",
    "Write a Python function to calculate fibonacci numbers",
    "Compare React and Vue.js frameworks",
]

# =============================================================================
# UI FEATURE FLAGS
# =============================================================================

UI_CONFIG = {
    "show_tier_badge": True,
    "show_cost_estimate": True,
    "enable_streaming": True,
    "max_history": 50,
}
