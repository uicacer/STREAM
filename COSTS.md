# Cost Configuration

## Single Source of Truth

Costs are defined in **`middleware/config.py`** in `MODEL_COSTS` dict.

## Syncing Costs

Costs must be kept in sync in TWO places:

1. **`middleware/config.py`** - STREAM uses this for calculations
2. **`gateway/litellm_config.yaml`** - LiteLLM uses this for database logging

**When updating costs:**
1. Update `middleware/config.py` -> `MODEL_COSTS`
2. Update `gateway/litellm_config.yaml` -> `model_info.input_cost_per_token` / `output_cost_per_token`
3. Restart middleware (it validates costs match on startup)

## Why Two Places?

- LiteLLM needs costs to log to PostgreSQL
- STREAM calculates costs in middleware
- Validation check ensures they stay in sync
