import logging
import os
import re
import subprocess

import httpx
import litellm

logger = logging.getLogger(__name__)


# url for Ollama model registry API
OLLAMA_REGISTRY_URL = "https://registry.ollama.ai/v2/library"


class OllamaModelManager:
    """Manage Ollama models with auto-pull capability"""

    def __init__(self):
        # Support both CLI (local dev) and HTTP API (Docker)
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        self.use_http = not self._is_ollama_cli_available()

        if self.use_http:
            logger.info(f"🐳 Using Ollama HTTP API at {self.ollama_url}")
        else:
            logger.info("💻 Using Ollama CLI")

        self.available_models = self._get_available_models()

    def _is_ollama_cli_available(self) -> bool:
        """Check if ollama CLI is available"""
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, timeout=2)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_available_models(self) -> set:
        """Get list of locally available models"""
        if self.use_http:
            # NEW: HTTP API approach (Docker)
            return self._get_available_models_http()
        else:
            # EXISTING: CLI approach (local dev)
            return self._get_available_models_cli()

    def _get_available_models_cli(self) -> set:
        """Get models via CLI (original method)"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            # Parse output - models are in first column
            models = set()
            for line in result.stdout.split("\n")[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    models.add(model_name)
            return models
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not check Ollama models: {e}")
            return set()

    def _get_available_models_http(self) -> set:
        """NEW: Get models via HTTP API (Docker)"""
        try:
            response = httpx.get(f"{self.ollama_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = set()
                for model in data.get("models", []):
                    models.add(model["name"])
                return models
            else:
                logger.warning(f"⚠️  Ollama API returned status {response.status_code}")
                return set()
        except Exception as e:
            logger.warning(f"⚠️  Could not connect to Ollama: {e}")
            return set()

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available locally"""
        if ":" in model_name:
            # User specified a specific tag - require exact match
            return model_name in self.available_models
        else:
            # User didn't specify tag - check if any version exists
            # Example: "llama3.2" matches "llama3.2:1b" or "llama3.2:3b"
            return any(
                m.startswith(model_name + ":") or m == model_name for m in self.available_models
            )

    def get_model_size_estimate(self, model_name: str) -> str:
        """
        Get model size using multiple strategies:
        1. Try Ollama registry API (most accurate)
        2. Parse parameter count from name (good estimate)
        3. Generic fallback
        """

        # Strategy 1: Try registry API (accurate but requires network)
        try:
            exact_size = self._fetch_size_from_registry(model_name)
            if exact_size:
                return exact_size + " (from registry)"
        except Exception:
            pass  # Fall through to estimation

        # Strategy 2: Parse parameter count (good estimate)
        estimated_size = self._estimate_from_parameters(model_name)
        if estimated_size:
            return estimated_size + " (estimated)"

        # Strategy 3: Generic fallback
        return "~2-5GB (size unknown)"

    def _fetch_size_from_registry(self, model_name: str) -> str | None:
        """Fetch actual size from Ollama registry"""
        try:
            if ":" in model_name:
                model, tag = model_name.split(":", 1)
            else:
                model = model_name
                tag = "latest"

            url = f"{OLLAMA_REGISTRY_URL}/{model}/manifests/{tag}"

            response = httpx.get(url, timeout=5.0)

            if response.status_code == 200:
                manifest = response.json()

                total_size = 0
                if "layers" in manifest:
                    for layer in manifest["layers"]:
                        total_size += layer.get("size", 0)

                size_gb = total_size / (1024**3)
                return f"~{size_gb:.1f}GB"

        except Exception:
            pass

        return None

    def _estimate_from_parameters(self, model_name: str) -> str | None:
        """Estimate size from parameter count in model name"""

        # Extract parameter count (e.g., "8b", "70b", "1b")
        match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())

        if match:
            params = float(match.group(1))

            """
                Estimation formula for quantized models.
                Based on typical Q4_K_M quantization:
                <2B models: ~1.3 GB/B (less optimization)
                2-10B models: ~0.6 GB/B (optimal quantization)
                10-20B models: ~0.7 GB/B (slightly larger)
                >20B models: ~0.6 GB/B (well-optimized)
            """
            if params < 2:
                size_gb = params * 1.3
            elif params < 10:
                size_gb = params * 0.6
            elif params < 20:
                size_gb = params * 0.7
            else:
                size_gb = params * 0.6

            return f"~{size_gb:.1f}GB ({params}B params)"

        # Check for size keywords
        name_lower = model_name.lower()
        if "tiny" in name_lower or "mini" in name_lower:
            return "~500MB-1GB"
        elif "small" in name_lower:
            return "~1-2GB"
        elif "medium" in name_lower:
            return "~3-5GB"
        elif "large" in name_lower:
            return "~7-10GB"

        return None

    def prompt_user_for_download(self, model_name: str) -> bool:
        """Ask user if they want to download the model (user-friendly)"""
        size_info = self.get_model_size_estimate(model_name)

        # Parse the size and source
        if "(from registry)" in size_info:
            size = size_info.split(" (from registry)")[0]
            source_explanation = "✓ Estimated from official catalog (actual size may vary by ~5%)"
        elif "(estimated)" in size_info:
            size = size_info.split(" (estimated)")[0]
            source_explanation = "~ Estimated based on model specifications"
        else:
            size = size_info
            source_explanation = "Exact size will appear during download"

        print("\n" + "=" * 70)
        print(f"📦 Download AI Model: {model_name}")
        print("=" * 70)
        print(f"  Size:       {size}")
        print(f"  Source:     {source_explanation}")
        print("  Storage:    Your computer (via Ollama)")
        print("  Duration:   ~2-10 minutes (depends on your internet speed)")
        print("=" * 70)

        response = input("Continue with download? [y/N]: ").lower().strip()
        return response in ["y", "yes"]

    def pull_model(self, model_name: str, show_progress: bool = True) -> bool:
        """Download an Ollama model"""
        if self.use_http:
            # NEW: HTTP API approach (Docker)
            return self._pull_model_http(model_name, show_progress)
        else:
            # EXISTING: CLI approach (local dev)
            return self._pull_model_cli(model_name, show_progress)

    def _pull_model_cli(self, model_name: str, show_progress: bool) -> bool:
        """EXISTING: Pull model via CLI"""
        print(f"\n📥 Downloading {model_name}...")
        print("This may take a few minutes depending on your connection.")
        print("You can cancel with Ctrl+C if needed.\n")

        try:
            # Run with or without progress output
            if show_progress:
                # Show real-time progress
                process = subprocess.Popen(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                # Stream output
                for line in process.stdout:
                    print(line, end="")

                process.wait()
                success = process.returncode == 0
            else:
                success = True

            if success:
                print(f"\n✅ Model '{model_name}' downloaded successfully!")
                self.available_models.add(model_name)
                return True
            else:
                print(f"\n❌ Failed to download '{model_name}'")
                return False

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error downloading model: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Download cancelled by user")
            return False

    def _pull_model_http(self, model_name: str, show_progress: bool) -> bool:
        """NEW: Pull model via HTTP API (Docker)"""
        print(f"\n📥 Downloading {model_name}...")
        print("This may take a few minutes depending on your connection.")
        print("You can cancel with Ctrl+C if needed.\n")

        try:
            # Start the pull request
            response = httpx.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                timeout=600.0,  # 10 minutes
            )

            if response.status_code == 200:
                print(f"\n✅ Model '{model_name}' downloaded successfully!")
                self.available_models.add(model_name)
                return True
            else:
                print(f"\n❌ Failed to download '{model_name}': {response.text}")
                return False

        except Exception as e:
            print(f"\n❌ Error downloading model: {e}")
            return False

    def ensure_model(
        self, model_name: str, auto_download: bool = False, silent: bool = False
    ) -> bool:
        """
        Ensure model is available, optionally auto-downloading

        Args:
            model_name: Name of the model (e.g., "llama2")
            auto_download: If True, download without prompting
            silent: If True, don't print status messages

        Returns:
            True if model is available, False otherwise
        """
        # Check if already available
        if self.is_model_available(model_name):
            if not silent:
                print(f"✅ Model '{model_name}' is available")
            return True

        if not silent:
            print(f"⚠️  Model '{model_name}' not found locally")

        # Auto-download or prompt
        if auto_download:
            if not silent:
                print(f"🔄 Auto-downloading '{model_name}'...")
            return self.pull_model(model_name, show_progress=not silent)
        else:
            # Ask user
            if self.prompt_user_for_download(model_name):
                return self.pull_model(model_name)
            else:
                if not silent:
                    print("❌ Skipping model download")
                return False

    def safe_completion(self, model_name: str, messages: list, **kwargs) -> litellm.ModelResponse:
        """
        Wrapper that ensures model exists before calling LiteLLM

        This is now a METHOD of the class!
        """
        if not self.ensure_model(model_name):
            raise Exception(f"Model {model_name} not available and user declined download")

        return litellm.completion(model=f"ollama/{model_name}", messages=messages, **kwargs)
