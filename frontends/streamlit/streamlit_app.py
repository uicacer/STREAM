# =============================================================================
# STREAM - Streamlit Chat Interface
# =============================================================================
# Simple, clean chat interface for STREAM
# =============================================================================

import logging
import os
import subprocess
import time
import traceback

import httpx
import streamlit as st
from config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    EXAMPLE_QUERIES,
)

from stream.sdk.python.chat_handler import ChatHandler

# Configure logging - use INFO level to reduce noise
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Silence noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# GLOBUS AUTHENTICATION HELPER
# =============================================================================


def restart_lakeshore_proxy() -> tuple[bool, str]:
    """
    Restart the lakeshore-proxy Docker container.

    This is needed when the container was started before credentials existed,
    because Docker's volume mount doesn't pick up files created after container start.

    Returns:
        (success: bool, message: str)
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info("🔄 Restarting lakeshore-proxy container...")

        # Get project root directory (parent of frontends/streamlit)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Try docker-compose first (preferred)
        result = subprocess.run(
            ["docker-compose", "restart", "lakeshore-proxy"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )

        if result.returncode == 0:
            logger.info("✅ Container restarted successfully")
            time.sleep(3)  # Wait for container to be healthy
            return True, "Container restarted successfully"

        # Try docker directly as fallback
        result = subprocess.run(
            ["docker", "restart", "stream-lakeshore-proxy"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("✅ Container restarted successfully (via docker)")
            time.sleep(3)
            return True, "Container restarted successfully"

        logger.error(f"❌ Failed to restart container: {result.stderr}")
        return False, f"Failed to restart: {result.stderr}"

    except subprocess.TimeoutExpired:
        logger.error("❌ Container restart timed out")
        return False, "Container restart timed out"
    except FileNotFoundError:
        logger.error("❌ docker-compose not found")
        return False, "docker-compose not found"
    except Exception as e:
        logger.error(f"❌ Failed to restart container: {e}")
        return False, f"Failed to restart: {str(e)}"


def reload_proxy_credentials() -> tuple[bool, str]:
    """
    Tell the lakeshore-proxy Docker container to reload Globus credentials.

    After the user authenticates on the host machine, the credentials are saved
    to ~/.globus_compute/storage.db. The Docker container needs to reload these
    credentials because it caches the GlobusApp state.

    If reload fails because the credentials file doesn't exist in the container
    (happens when container was started before auth), automatically restart
    the container to pick up the newly created file.

    Returns:
        (success: bool, message: str)
    """
    logger = logging.getLogger(__name__)
    proxy_url = os.getenv("LAKESHORE_PROXY_URL", "http://localhost:8001")

    try:
        logger.info(f"🔄 Reloading proxy credentials at {proxy_url}/reload-auth")

        with httpx.Client(timeout=10.0) as client:
            response = client.post(f"{proxy_url}/reload-auth")

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info(f"✅ Proxy credentials reloaded: {result.get('message')}")
                    return True, result.get("message", "Credentials reloaded")

                error_msg = result.get("message", "Reload failed")
                logger.warning(f"⚠️ Proxy reload reported failure: {error_msg}")

                # Check if this is the "database file not found" error
                # This happens when container was started before credentials existed
                if "unable to open database file" in error_msg.lower():
                    logger.info(
                        "🔄 Credentials file not visible to container, restarting container..."
                    )
                    restart_success, restart_msg = restart_lakeshore_proxy()
                    if restart_success:
                        # Try reload again after restart
                        logger.info("🔄 Retrying credential reload after container restart...")
                        response2 = client.post(f"{proxy_url}/reload-auth")
                        if response2.status_code == 200:
                            result2 = response2.json()
                            if result2.get("success"):
                                logger.info("✅ Credentials reloaded after container restart!")
                                return True, "Credentials reloaded (container was restarted)"
                        # Even if reload fails after restart, the container should now see the file
                        return True, "Container restarted - credentials should be available"

                    return False, f"Container restart failed: {restart_msg}"

                return False, error_msg

            logger.error(f"❌ Proxy reload failed with status {response.status_code}")
            return False, f"Proxy returned status {response.status_code}"

    except httpx.ConnectError:
        # Proxy not running - might be okay if using SSH mode
        logger.warning("Could not connect to lakeshore-proxy (might not be running)")
        return True, "Proxy not reachable (may be using SSH mode)"
    except Exception as e:
        logger.error(f"❌ Failed to reload proxy credentials: {e}")
        return False, f"Failed to reload: {str(e)}"


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
    9. **NEW**: Tells the Docker proxy to reload the credentials

    This is TRUE zero-friction - you only interact with the browser,
    no terminal commands or code pasting required!

    Returns:
        (success: bool, message: str)
    """
    logger = logging.getLogger(__name__)

    try:
        # First check if already authenticated
        from stream.middleware.core.globus_auth import is_authenticated

        if is_authenticated():
            logger.info("✓ Already authenticated with Globus Compute")
            # Still reload proxy credentials in case container has stale state
            reload_proxy_credentials()
            return True, "✅ Already authenticated! Your credentials are valid."

        # Import our custom zero-friction OAuth implementation
        from stream.middleware.core.globus_auth import authenticate_with_browser_callback

        logger.info("Starting Globus authentication flow...")

        # This handles the entire OAuth flow automatically
        # It will:
        # - Open your browser to Globus auth page
        # - Start a local server to receive the callback
        # - Capture the auth code automatically
        # - Exchange it for tokens
        # - Save the tokens to disk
        success, message = authenticate_with_browser_callback()

        if success:
            logger.info(f"Authentication successful: {message}")

            # IMPORTANT: Tell the Docker proxy to reload credentials
            # The proxy caches the GlobusApp state, so it needs to be notified
            # that new credentials are available
            reload_success, reload_msg = reload_proxy_credentials()
            if reload_success:
                logger.info(f"Proxy notified: {reload_msg}")
            else:
                logger.warning(f"Proxy notification failed: {reload_msg}")
                # Don't fail overall auth - credentials are saved, just proxy needs restart
        else:
            logger.error(f"Authentication failed: {message}")

        return success, message

    except ImportError as e:
        # Fallback: If our custom module isn't available, explain the issue
        error_details = traceback.format_exc()
        logger.error(f"Import error: {error_details}")
        return False, f"❌ Zero-friction auth module not available: {str(e)}\n{error_details}"

    except Exception as e:
        # Any other error during authentication
        error_details = traceback.format_exc()
        logger.error(f"Authentication error: {error_details}")
        return False, f"❌ Authentication failed: {str(e)}\n{error_details}"


MAX_DISPLAY_MESSAGES = 100  # UI display limit
MAX_SEND_MESSAGES = 50  # Messages sent to middleware (middleware will trim further if needed)

# =============================================================================
# ENGAGING PROGRESS MESSAGES
# =============================================================================

# Tier-specific status messages (simplified for faster rendering)
TIER_MESSAGES = {
    "local": {
        "icon": "🏠",
        "name": "Local AI",
        "status": "Generating response...",
    },
    "lakeshore": {
        "icon": "🏫",
        "name": "Lakeshore HPC",
        "status": "Generating response...",
    },
    "cloud": {
        "icon": "☁️",
        "name": "Cloud AI",
        "status": "Generating response...",
    },
    "auto": {
        "icon": "🤖",
        "name": "Smart Router",
        "status": "Routing and generating...",
    },
}


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
        "lakeshore_queries": 0,
        "cloud_queries": 0,
        "total_cost": 0.0,
    }

if "judge_strategy" not in st.session_state:
    st.session_state.judge_strategy = "ollama-3b"  # Default

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

        # Judge Strategy Selector (only enabled in Auto mode)
        st.markdown("---")
        st.markdown("**🧠 Complexity Judge**")

        # Check if Auto mode is selected
        is_auto_mode = st.session_state.tier_preference == "auto"

        if is_auto_mode:
            st.caption("_Choose how STREAM analyzes query complexity_")
        else:
            st.caption("_Disabled when using a specific tier_")

        judge_options = {
            "🎯 Ollama 3b (Balanced, free)": "ollama-3b",
            "👁️ Gemma Vision (Multimodal, free)": "gemma-vision",
            "🚀 Claude Haiku (~$1/5K judgments)": "haiku",
        }

        selected_judge = st.radio(
            "Judge Strategy",
            options=list(judge_options.keys()),
            index=1,  # Default to Ollama 3b
            help="Choose how STREAM analyzes query complexity. Only used in Auto mode.",
            label_visibility="collapsed",
            disabled=not is_auto_mode,  # Disable when specific tier is selected
        )

        st.session_state.judge_strategy = judge_options[selected_judge]

        # Show info about current selection (only when enabled)
        if is_auto_mode:
            if st.session_state.judge_strategy == "haiku":
                st.info(
                    "💰 Claude Haiku costs approximately **$1 per 5,000 query judgments**. Most accurate option."
                )
            elif st.session_state.judge_strategy == "gemma-vision":
                st.info(
                    "🆓 Completely free! Vision-capable judge that can assess image complexity. Uses Gemma 3 4B."
                )
            else:
                st.info(
                    "🆓 Completely free! Speed depends on your machine's hardware. Good balance of speed and accuracy."
                )

    st.divider()

    # Session Stats
    st.subheader("📊 Session Stats")

    with st.expander("ℹ️ Context Window Limits"):
        st.caption("""
        **Per-tier context limits:**
        - 💻 LOCAL: ~8,000 tokens
        - 🏫 LAKESHORE: ~8,000 tokens
        - ☁️ CLOUD: ~200,000 tokens

        _STREAM automatically routes to larger tiers when needed._
        """)

    stats = st.session_state.session_stats

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", stats["queries"])
        st.metric("Total Cost", f"${stats['total_cost']:.4f}")
    with col2:
        st.metric("💻 Local", stats["local_queries"], delta="Free")
        st.metric("🏫 Lakeshore", stats.get("lakeshore_queries", 0), delta="Low cost")
        st.metric("☁️ Cloud", stats["cloud_queries"])

    st.caption(
        "_💡 Costs are estimated from token usage. Local is free, Lakeshore is minimal cost._"
    )

    st.divider()

    # Actions
    st.subheader("🎯 Actions")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_handler.clear_history()
        st.session_state.session_stats = {
            "queries": 0,
            "local_queries": 0,
            "lakeshore_queries": 0,
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
                model_display = "GPT-4o"
                tier_emoji = "🤖"  # OpenAI
            elif "4o-mini" in model.lower() or model == "cloud-gpt-cheap":
                model_display = "GPT-4o Mini"
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
            if tier == "local":
                st.caption("💰 FREE")
            elif tier == "lakeshore":
                if cost > 0:
                    st.caption(f"💰 ~${cost:.6f}")
                else:
                    st.caption("💰 Low cost")
            elif tier == "cloud":
                st.caption(f"💰 ~${cost:.6f}")
            else:
                st.caption("💰 FREE")

# Chat input
if prompt := st.chat_input("Ask me anything about HPC, SLURM, or general questions..."):
    st.session_state.pending_query = prompt

# =============================================================================
# HANDLE ONGOING AUTH FLOW (persists across reruns)
# =============================================================================
# Only require auth_flow_step to be set - auth_pending_message is optional
if st.session_state.get("auth_flow_step"):
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

            # VPN Warning - important for successful authentication
            st.warning(
                "⚠️ **VPN Users**: Please **disconnect your VPN** before clicking "
                '"Authenticate Now". Some VPNs interfere with the authentication callback. '
                "You can reconnect after authentication completes."
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
                "### 🌐 Authenticating with Globus Compute\n\n"
                "Please check your browser - a Globus login page should have opened.\n\n"
                "**If no browser opened**, you may already be authenticated!"
            )

            # Show spinner while authenticating
            with st.spinner(
                "🔄 Authenticating with Globus Compute... Please complete login in your browser"
            ):
                success, message = authenticate_globus_compute()

            if success:
                st.success(f"✅ {message}")
                import time

                time.sleep(1)  # Brief pause to show success message
                st.session_state.auth_flow_step = "vpn_reconnect"
                st.rerun()
            else:
                st.error(f"❌ {message}")
                st.session_state.auth_flow_step = "vpn_warning"
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Try Again", type="primary", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("☁️ Use Cloud Instead", type="secondary", use_container_width=True):
                        st.session_state.tier_preference = "cloud"
                        st.session_state._sync_tier_selector = True
                        st.session_state.auth_flow_step = None
                        st.session_state.pop("auth_pending_message", None)
                        st.rerun()
                st.stop()

        # STEP 3: Success - auto-retry the question
        elif st.session_state.auth_flow_step == "vpn_reconnect":
            st.success(
                "### ✅ Authentication Successful!\n\n"
                "You can now reconnect your VPN if needed. "
                "Your credentials are cached and will work through VPN."
            )

            import time

            time.sleep(1)  # Brief pause to show success

            # Auto-retry the question if there was one
            st.session_state.auth_flow_step = None
            if user_message:
                st.info(
                    f"📝 Retrying your question: *{user_message[:50]}...*"
                    if len(user_message) > 50
                    else f"📝 Retrying your question: *{user_message}*"
                )
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
        # Only pass judge_strategy when in Auto mode
        judge_strategy = (
            st.session_state.judge_strategy if st.session_state.tier_preference == "auto" else None
        )
        result = st.session_state.chat_handler.chat(
            user_message,
            user_preference=st.session_state.tier_preference,
            stream=True,  # ENABLE STREAMING
            temperature=temperature,
            judge_strategy=judge_strategy,
        )

        if result["success"]:
            full_response = ""
            start_time = time.time()
            first_chunk_received = False

            # Get tier-specific info for status messages
            tier_pref = st.session_state.tier_preference
            tier_info = TIER_MESSAGES.get(tier_pref, TIER_MESSAGES["auto"])

            # Simple spinner - minimal overhead before streaming starts
            status_container = st.status(
                f"{tier_info['icon']} {tier_info['status']}",
                expanded=False,  # Collapsed by default - less visual noise
            )

            # Stream response
            try:
                for chunk in result["response"]:
                    if not first_chunk_received:
                        # First chunk arrived! Get the actual tier and complexity from metadata
                        actual_meta = st.session_state.chat_handler.get_last_stream_metadata()
                        actual_tier = actual_meta.get("tier", "unknown")
                        actual_complexity = actual_meta.get("complexity", "unknown")
                        is_fallback = actual_meta.get("fallback_used", False)
                        original_tier = actual_meta.get("original_tier")

                        # Tier display info
                        tier_icons = {
                            "local": {"icon": "💻", "name": "Local"},
                            "lakeshore": {"icon": "🏫", "name": "Lakeshore"},
                            "cloud": {"icon": "☁️", "name": "Cloud"},
                        }
                        tier_info_display = tier_icons.get(
                            actual_tier, {"icon": "🤖", "name": actual_tier.title()}
                        )

                        # Complexity display (use actual complexity from middleware)
                        complexity_display = {
                            "low": "Low complexity",
                            "medium": "Medium complexity",
                            "high": "High complexity",
                        }
                        complexity_desc = complexity_display.get(actual_complexity, "")

                        # Show routing decision in status
                        if is_fallback and original_tier:
                            # Fallback occurred - show original tier that failed
                            original_name = tier_icons.get(original_tier, {}).get(
                                "name", original_tier.title()
                            )
                            label = f"🔄 {complexity_desc} → {tier_info_display['name']} ({original_name} unavailable)"
                        elif tier_pref == "auto" and complexity_desc:
                            label = f"{tier_info_display['icon']} {complexity_desc} → {tier_info_display['name']}"
                        else:
                            label = f"{tier_info_display['icon']} {tier_info_display['name']}"

                        status_container.update(
                            label=label,
                            state="complete",
                            expanded=False,
                        )
                        first_chunk_received = True
                        full_response = chunk
                        message_placeholder.markdown(full_response + "▌")
                    else:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

            except Exception as e:
                status_container.update(
                    label="❌ Stream interrupted",
                    state="error",
                    expanded=False,
                )
                st.error(f"Stream interrupted: {e}")
                # Save partial response
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response + " [INTERRUPTED]"}
                )

            # Ensure status is marked complete (if we got here without chunks)
            if not first_chunk_received:
                status_container.update(
                    label=f"{tier_info['icon']} Waiting...",
                    state="running",
                )

            # Show final response or placeholder
            if first_chunk_received:
                message_placeholder.markdown(full_response)
            else:
                # Check for auth error BEFORE displaying empty response
                stream_meta = st.session_state.chat_handler.get_last_stream_metadata()
                if not stream_meta.get("auth_required"):
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

                # ========== SHOW JUDGE FALLBACK NOTIFICATION ==========
                judge_fallback = stream_meta.get("judge_fallback")
                if judge_fallback:
                    method = judge_fallback.get("method", "")
                    if method == "keyword_fallback":
                        st.info(
                            "ℹ️ **Smart routing used keyword matching** - "
                            "The LLM judge was unavailable, so routing was based on keywords in your query.",
                            icon="🔤",
                        )
                    elif method == "default_fallback":
                        st.info(
                            "ℹ️ **Smart routing used default (MEDIUM)** - "
                            "The LLM judge was unavailable and no keywords matched. "
                            "Your query was routed to the Lakeshore tier.",
                            icon="📊",
                        )
                # ========== END JUDGE FALLBACK NOTIFICATION ==========

                # Get cost from middleware (single source of truth)
                # If cost is 0 for cloud/lakeshore, that's a middleware issue to fix there
                cost = stream_meta.get("cost", 0.0)

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
                elif tier == "lakeshore":
                    st.session_state.session_stats["lakeshore_queries"] += 1
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
                            "lakeshore_queries": 0,
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
