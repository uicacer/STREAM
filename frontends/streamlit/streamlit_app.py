# =============================================================================
# STREAM - Streamlit Chat Interface
# =============================================================================
# Simple, clean chat interface for STREAM
# =============================================================================

import logging
import time

import streamlit as st
from config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    EXAMPLE_QUERIES,
)

from stream.sdk.python.chat_handler import ChatHandler

# Configure logging to see SDK debug messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# =============================================================================
# GLOBUS AUTHENTICATION HELPER
# =============================================================================


def authenticate_globus_compute() -> tuple[bool, str]:
    """
    Zero-friction Globus Compute authentication using automatic OAuth callback.

    This function implements a seamless authentication flow:
    1. Checks if already authenticated (returns immediately if so)
    2. Starts a local OAuth callback server on localhost:8765
    3. Opens your browser to the Globus login page
    4. You authenticate in the browser (just click "Allow")
    5. Globus redirects back to our local server
    6. The OAuth code is captured automatically (no manual copying!)
    7. Code is exchanged for access tokens
    8. Tokens are saved to ~/.globus_compute/ for future use

    This is TRUE zero-friction - you only interact with the browser,
    no terminal commands or code pasting required!

    Returns:
        (success: bool, message: str)
    """
    try:
        # Import our custom zero-friction OAuth implementation
        from stream.middleware.core.globus_auth import authenticate_with_browser_callback

        # This handles the entire OAuth flow automatically
        # It will:
        # - Open your browser to Globus auth page
        # - Start a local server to receive the callback
        # - Capture the auth code automatically
        # - Exchange it for tokens
        # - Save the tokens to disk
        success, message = authenticate_with_browser_callback()
        return success, message

    except ImportError as e:
        # Fallback: If our custom module isn't available, explain the issue
        import traceback

        error_details = traceback.format_exc()
        return False, f"❌ Zero-friction auth module not available: {str(e)}\n{error_details}"

    except Exception as e:
        # Any other error during authentication
        import traceback

        error_details = traceback.format_exc()
        return False, f"❌ Authentication failed: {str(e)}\n{error_details}"


MAX_DISPLAY_MESSAGES = 100  # UI display limit
MAX_SEND_MESSAGES = 50  # Messages sent to middleware (middleware will trim further if needed)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "chat_handler" not in st.session_state:
    st.session_state.chat_handler = ChatHandler()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tier_preference" not in st.session_state:
    st.session_state.tier_preference = "auto"

if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "queries": 0,
        "local_queries": 0,
        "cloud_queries": 0,
        "total_cost": 0.0,
    }

if "last_actual_tier" not in st.session_state:
    st.session_state.last_actual_tier = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)

    st.divider()

    # Tier Selection
    st.subheader("⚙️ Settings")

    tier_options = {
        "🤖 Auto (Smart Routing)": "auto",
        "💻 Local (Ollama - Free)": "local",
        "🏫 Lakeshore (Campus GPU)": "lakeshore",
        "☁️ Cloud (Claude/GPT - Paid)": "cloud",
    }
    tier_labels = list(tier_options.keys())
    tier_values = list(tier_options.values())

    # Only sync selectbox when there was a programmatic tier change (flag set by buttons)
    if st.session_state.get("_sync_tier_selector"):
        current_pref = st.session_state.tier_preference
        if current_pref in tier_values:
            st.session_state.tier_selector = tier_labels[tier_values.index(current_pref)]
        st.session_state._sync_tier_selector = False  # Clear the flag

    selected_tier = st.selectbox(
        "Model Tier",
        options=tier_labels,
        help="Auto: Let STREAM decide based on query complexity",
        key="tier_selector",
    )

    # Update tier_preference from selectbox (this is the user's choice)
    st.session_state.tier_preference = tier_options[selected_tier]

    # Advanced Settings (collapsed by default)
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused",
        )

        show_routing = st.checkbox(
            "Show Routing Details", value=True, help="Display which tier handled each query"
        )

    st.divider()

    # Session Stats
    st.subheader("📊 Session Stats")

    with st.expander("ℹ️ Context Window Limits"):
        st.caption("""
        **Per-tier context limits:**
        - 💻 LOCAL: ~2,000 tokens
        - 🏫 LAKESHORE: ~8,000 tokens
        - ☁️ CLOUD: ~200,000 tokens

        _STREAM automatically routes to larger tiers when needed._
        """)

    stats = st.session_state.session_stats

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", stats["queries"])
        st.metric("Local", stats["local_queries"], delta="Free")
    with col2:
        st.metric("Cloud", stats["cloud_queries"])
        st.metric("Total Cost", f"${stats['total_cost']:.4f}")

    st.caption("_💡 Cloud costs are estimated (~4 char/token). Actual costs may vary._")

    st.divider()

    # Actions
    st.subheader("🎯 Actions")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_handler.clear_history()
        st.session_state.session_stats = {
            "queries": 0,
            "local_queries": 0,
            "cloud_queries": 0,
            "total_cost": 0.0,
        }
        st.rerun()

    if st.button("📋 Export History", use_container_width=True):
        history = st.session_state.chat_handler.export_history()
        st.download_button(
            label="💾 Download JSON",
            data=str(history),
            file_name="stream_chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # Example Queries
    st.subheader("💡 Try These")

    for example in EXAMPLE_QUERIES[:3]:
        if st.button(example, use_container_width=True, key=f"example_{example}"):
            st.session_state.pending_query = example
            st.rerun()

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

st.title(f"{APP_ICON} STREAM Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # SHOW FALLBACK WARNING FOR SAVED MESSAGES
    if (
        message["role"] == "assistant"
        and "metadata" in message
        and message["metadata"].get("fallback_used")
    ):
        original_tier = message["metadata"].get("original_tier", "unknown")
        current_tier = message["metadata"].get("tier", "unknown")
        st.warning(
            f"⚠️ **Tier Fallback**: {original_tier.upper()} was unavailable, "
            f"automatically switched to {current_tier.upper()}.",
            icon="🔄",
        )

    # Show routing info if enabled and message is from assistant
    if show_routing and message["role"] == "assistant" and "metadata" in message:
        meta = message["metadata"]
        tier = meta["tier"]
        model = meta.get("model", "unknown")

        # Determine emoji and display text
        if tier == "local":
            tier_emoji = "💻"
            tier_display = "LOCAL"
            # Parse model name: "local-llama" -> "Llama"
            if "tiny" in model:
                model_display = "Llama 3.2 1B"
            elif "quality" in model:
                model_display = "Llama 3.1 8B"
            else:
                model_display = "Llama 3.2 3B"

        elif tier == "lakeshore":
            tier_emoji = "🏫"
            tier_display = "LAKESHORE"
            model_display = "vLLM"

        else:  # cloud
            tier_emoji = "☁️"
            tier_display = "CLOUD"
            # Show specific cloud provider
            if "claude" in model.lower():
                model_display = "Claude Sonnet 4"
                tier_emoji = "🔮"  # Anthropic
            elif "gpt-4" in model.lower():
                model_display = "GPT-4 Turbo"
                tier_emoji = "🤖"  # OpenAI
            elif "gpt-3.5" in model.lower():
                model_display = "GPT-3.5"
                tier_emoji = "⚡"  # OpenAI budget
            else:
                model_display = model

        # Display with better formatting
        col1, col2, col3 = st.columns([3, 2, 5])
        with col1:
            st.caption(f"{tier_emoji} **{tier_display}** · {model_display}")
        with col2:
            duration = meta.get("duration", 0)
            st.caption(f"⏱️ {duration:.2f}s")
        with col3:
            cost = meta.get("cost", 0)
            if tier in ["cloud", "lakeshore"] and cost > 0:
                st.caption(f"💰 ~${cost:.6f} (estimate)")
            elif tier == "lakeshore" and cost == 0:
                st.caption("💰 Campus GPU (low cost)")
            else:
                st.caption("💰 FREE")

# Chat input
if prompt := st.chat_input("Ask me anything about HPC, SLURM, or general questions..."):
    st.session_state.pending_query = prompt

# =============================================================================
# HANDLE ONGOING AUTH FLOW (persists across reruns)
# =============================================================================
if st.session_state.get("auth_flow_step") and st.session_state.get("auth_pending_message"):
    # Auth flow is in progress - handle it here
    user_message = st.session_state.get("auth_pending_message", "")

    with st.chat_message("assistant"):
        # STEP 1: Authentication prompt
        if st.session_state.auth_flow_step == "vpn_warning":
            st.info(
                "### 🔐 Globus Compute Authentication Required\n\n"
                "To use the **Lakeshore HPC tier**, you need to authenticate with Globus Compute.\n\n"
                "A browser window will open for you to log in. This is a **one-time setup** - "
                "your credentials will be saved for future sessions."
            )

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "🔐 Authenticate Now",
                    type="primary",
                    use_container_width=True,
                    key="auth_start",
                ):
                    st.session_state.auth_flow_step = "authenticating"
                    st.rerun()
            with col2:
                if st.button(
                    "☁️ Skip - Use Cloud",
                    use_container_width=True,
                    type="secondary",
                    key="auth_skip",
                ):
                    st.session_state.tier_preference = "cloud"
                    st.session_state._sync_tier_selector = True  # Sync the dropdown
                    st.session_state.auth_flow_step = None
                    if user_message:
                        st.session_state.pending_query = user_message
                    st.session_state.pop("auth_pending_message", None)
                    st.rerun()

            st.stop()

        # STEP 2: Perform Authentication
        elif st.session_state.auth_flow_step == "authenticating":
            st.info(
                "### 🌐 Opening Browser for Authentication\n\nA browser window will open automatically..."
            )

            with st.spinner("Authenticating with Globus Compute..."):
                success, message = authenticate_globus_compute()

            if success:
                st.session_state.auth_flow_step = "vpn_reconnect"
                st.rerun()
            else:
                st.error(f"Authentication failed: {message}")
                st.session_state.auth_flow_step = "vpn_warning"
                if st.button("🔄 Try Again", type="primary"):
                    st.rerun()
                st.stop()

        # STEP 3: Success - auto-retry the question
        elif st.session_state.auth_flow_step == "vpn_reconnect":
            st.success("### ✅ Authentication Successful!")
            st.info("Retrying your question...")

            # Auto-retry the question
            st.session_state.auth_flow_step = None
            if user_message:
                st.session_state.pending_query = user_message
            st.session_state.pop("auth_pending_message", None)
            st.rerun()

# Handle pending query (from input or example button)
if "pending_query" in st.session_state:
    user_message = st.session_state.pending_query
    del st.session_state.pending_query

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    # Get response from chat handler
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Get streaming response
        result = st.session_state.chat_handler.chat(
            user_message,
            user_preference=st.session_state.tier_preference,
            stream=True,  # ENABLE STREAMING
            temperature=temperature,
        )

        if result["success"]:
            full_response = ""
            start_time = time.time()

            # Show spinner until first chunk arrives
            with st.spinner("Generating response..."):
                first_chunk = next(result["response"], None)
                if first_chunk:
                    full_response += first_chunk

            # Stream remaining chunks
            if first_chunk:
                message_placeholder.markdown(full_response + "▌")

                try:
                    for chunk in result["response"]:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                except Exception as e:
                    st.error(f"Stream interrupted: {e}")
                    # Save partial response
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response + " [INTERRUPTED]"}
                    )

                message_placeholder.markdown(full_response)

            # Check for auth error BEFORE displaying empty response
            stream_meta = st.session_state.chat_handler.get_last_stream_metadata()
            if stream_meta.get("auth_required"):
                # Auth required - don't show empty response, show auth flow instead
                pass  # Will be handled below
            elif not first_chunk:
                # No auth error but empty response - show placeholder
                message_placeholder.markdown("*No response received.*")

            # Get metadata from completed stream
            stream_meta = st.session_state.chat_handler.get_last_stream_metadata()

            # ========== CHECK FOR AUTHENTICATION ERROR IN STREAM METADATA ==========
            # Use structured error flag instead of parsing emoji/strings
            if stream_meta.get("auth_required"):
                # Set up auth flow state and rerun - the outer auth handler will display the UI
                st.session_state.auth_flow_step = "vpn_warning"
                st.session_state.auth_pending_message = user_message
                message_placeholder.empty()
                # Remove the user message from chat history since we'll retry after auth
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
                st.rerun()

            # ========== END AUTH ERROR CHECK ==========
            else:
                # No auth error - proceed with normal message handling
                tier = stream_meta.get("tier", "unknown")
                model = stream_meta.get("model", "unknown")
                duration = time.time() - start_time

                # ========== SHOW SLOW RESPONSE WARNING FOR LAKESHORE ==========
                if tier == "lakeshore" and duration > 15:
                    st.warning(
                        f"⏱️ **Response took {duration:.0f}s** - The Lakeshore HPC endpoint may have "
                        f"experienced a temporary issue and retried automatically.",
                        icon="🔄",
                    )
                # ========== END SLOW RESPONSE WARNING ==========

                # ========== SHOW FALLBACK WARNING IF IT HAPPENED ==========
                if stream_meta.get("fallback_used"):
                    original_tier = stream_meta.get("original_tier", "unknown")
                    st.warning(
                        f"⚠️ **Tier Fallback**: {original_tier.upper()} was unavailable, "
                        f"automatically switched to {tier.upper()}.",
                        icon="🔄",
                    )
                # ========== END FALLBACK WARNING ==========

                # **SHOW "CALCULATING..." FOR COST**
                cost = stream_meta.get("cost", 0.0)
                correlation_id = result.get("correlation_id")

                # **ESTIMATE COST FOR STREAMING**
                # Streaming doesn't return usage tokens, so we estimate based on 4 char/token
                if cost == 0.0 and tier in ["cloud", "lakeshore"]:
                    # Get last user message length
                    user_messages = [
                        m for m in st.session_state.messages if m.get("role") == "user"
                    ]
                    input_chars = len(user_messages[-1].get("content", "")) if user_messages else 0
                    output_chars = len(full_response)

                    # Estimate tokens: ~4 characters per token
                    input_tokens = input_chars // 4
                    output_tokens = output_chars // 4

                    # Calculate cost using rates from middleware (single source of truth)
                    cost = st.session_state.chat_handler.estimate_cost(
                        model, input_tokens, output_tokens
                    )

                # Add assistant message to chat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "metadata": {
                            "tier": tier,
                            "model": model,
                            "duration": duration,
                            "cost": cost,
                            "fallback_used": stream_meta.get("fallback_used", False),
                            "original_tier": stream_meta.get("original_tier"),
                        },
                    }
                )

                # Track actual tier used
                st.session_state.last_actual_tier = tier

                # Update session stats
                st.session_state.session_stats["queries"] += 1
                if tier == "local":
                    st.session_state.session_stats["local_queries"] += 1
                elif tier == "cloud":
                    st.session_state.session_stats["cloud_queries"] += 1

                # Accumulate cost
                st.session_state.session_stats["total_cost"] += cost

        else:
            # Show error
            error_msg = result.get("error", "Unknown error")

            # ========== USE STRUCTURED ERROR FLAGS (NOT EMOJI/STRING PARSING) ==========
            # Check if it's an authentication error using explicit flag
            is_auth_error = (
                result.get("auth_required") or result.get("error_type") == "authentication"
            )

            # Check if it's a context window error using error type
            is_context_error = result.get("error_type") == "context_too_long"
            # ========== END STRUCTURED ERROR CHECK ==========

            if is_auth_error:
                # Use the same VPN-aware auth flow as the streaming case
                st.session_state.auth_flow_step = "vpn_warning"
                st.session_state.auth_pending_message = user_message
                message_placeholder.empty()
                # Remove the user message from chat history since we'll retry after auth
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
                st.rerun()

            elif is_context_error:
                message_placeholder.error("💬 **Conversation Too Long**")
                st.warning(
                    "This conversation has exceeded the model's context window. "
                    "Please start a new conversation or switch to Cloud tier for longer chats."
                )

                # Show clear chat button prominently
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "🗑️ Start New Conversation", type="primary", use_container_width=True
                    ):
                        st.session_state.messages = []
                        st.session_state.chat_handler.clear_history()
                        st.session_state.last_actual_tier = None  # ← Reset this too
                        st.session_state.session_stats = {
                            "queries": 0,
                            "local_queries": 0,
                            "cloud_queries": 0,
                            "total_cost": 0.0,
                        }
                        st.rerun()
                with col2:
                    if st.button("☁️ Use Cloud Instead", type="secondary", use_container_width=True):
                        st.session_state.tier_preference = "cloud"
                        st.session_state._sync_tier_selector = True  # Sync the dropdown
                        st.rerun()
            else:
                message_placeholder.error(f"❌ Error: {error_msg}")

                # ========== ADD HELPFUL MESSAGE FOR 400 ERRORS ==========
                if "400" in error_msg:
                    st.info(
                        "💡 This might be a context limit issue. Try starting a new conversation or using Cloud tier."
                    )
                # ========== END ADD ==========

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ Error: {error_msg}",
                        "metadata": {"tier": "error"},
                    }
                )

    st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🌊 STREAM v1.0")

with col2:
    st.caption("Built with Streamlit + LiteLLM")

with col3:
    gateway_status = "🟢 Online" if st.session_state.chat_handler else "🔴 Offline"
    st.caption(f"Gateway: {gateway_status}")
