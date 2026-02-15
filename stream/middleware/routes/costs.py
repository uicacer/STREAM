"""
Cost tracking and analytics endpoints.
"""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from stream.middleware.config import STREAM_MODE
from stream.middleware.core.database import get_database_pool, is_database_available
from stream.middleware.core.database_sqlite import get_cost_summary as get_cost_summary_sqlite
from stream.middleware.utils.cost_reader import load_model_pricing

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# RESPONSE MODELS (Pydantic)
# =============================================================================


class ModelCostInfo(BaseModel):
    """Pricing information for a single model."""

    input: float = Field(..., description="Cost per input token in USD", ge=0.0)
    output: float = Field(..., description="Cost per output token in USD", ge=0.0)


# =============================================================================
# MODEL PRICING ENDPOINT
# =============================================================================


@router.get("/costs/models")
async def get_model_costs():
    """Get pricing information for all available models."""
    logger.info("Fetching model cost information")

    # Read from LiteLLM config (single source of truth)
    # pricing = get_all_model_costs()
    pricing = load_model_pricing()

    return {
        "success": True,
        "costs": pricing,
        "source": "litellm_config.yaml",
        "timestamp": datetime.now(UTC).isoformat(),
    }


# =============================================================================
# USAGE SUMMARY ENDPOINT
# =============================================================================


@router.get("/costs/summary")
async def get_cost_summary(
    days: int = Query(default=7, ge=1, le=365, description="Number of days to look back"),
):
    """Get usage and cost summary for the specified time period."""

    # Check if database is available
    if not is_database_available():
        logger.error("Cost summary requested but database is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "database_unavailable",
                "message": "Cost tracking database is not available",
                "suggestion": "Check POSTGRES_* environment variables"
                if STREAM_MODE == "server"
                else "SQLite database failed to initialize",
            },
        )

    # -------------------------------------------------------------------------
    # DESKTOP MODE: Query from SQLite
    # -------------------------------------------------------------------------
    # In desktop mode, costs are stored in a local SQLite file (~/.stream/data/costs.db).
    # We use a dedicated query function that handles SQLite's different SQL syntax:
    #   - Parameter placeholders: ? instead of %s
    #   - Table name: spend_logs instead of "LiteLLM_SpendLogs"
    #   - No connection pool (SQLite uses a single shared connection)
    if STREAM_MODE == "desktop":
        return get_cost_summary_sqlite(days)

    # -------------------------------------------------------------------------
    # SERVER MODE: Query from PostgreSQL (existing behavior below)
    # -------------------------------------------------------------------------
    # Get pool from core.database
    db_pool = get_database_pool()

    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        start_date = datetime.now(UTC) - timedelta(days=days)

        logger.info(
            f"Querying cost summary: {days} days",
            extra={"days": days, "start_date": start_date.isoformat()},
        )

        cur.execute(
            """
            SELECT
                model,
                COUNT(*) as requests,
                SUM(spend) as total_cost,
                SUM(prompt_tokens) as input_tokens,
                SUM(completion_tokens) as output_tokens
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" >= %s
            GROUP BY model
            ORDER BY total_cost DESC
            """,
            (start_date,),
        )

        results = cur.fetchall()

        summary = {
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.now(UTC).isoformat(),
            "models": [],
            "total_cost": 0.0,
            "total_requests": 0,
        }

        for row in results:
            model_name = row[0]
            request_count = row[1]
            model_cost = float(row[2]) if row[2] else 0.0
            input_tokens = row[3] or 0
            output_tokens = row[4] or 0

            avg_cost = model_cost / request_count if request_count > 0 else 0.0

            model_data = {
                "model": model_name,
                "requests": request_count,
                "cost": model_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "avg_cost_per_request": avg_cost,
            }

            summary["models"].append(model_data)
            summary["total_cost"] += model_cost
            summary["total_requests"] += request_count

        summary["avg_cost_per_request"] = (
            summary["total_cost"] / summary["total_requests"]
            if summary["total_requests"] > 0
            else 0.0
        )

        cur.close()

        logger.info(
            f"Cost summary: ${summary['total_cost']:.2f} across {summary['total_requests']} requests",
            extra={
                "total_cost": summary["total_cost"],
                "total_requests": summary["total_requests"],
            },
        )

        return summary

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Database error: {str(e)}", exc_info=True, extra={"days": days})
        # FIX: Add exception chaining (B904)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "database_error",
                "message": f"Failed to query cost data: {str(e)}",
            },
        ) from e

    finally:
        if conn:
            db_pool.putconn(conn)
