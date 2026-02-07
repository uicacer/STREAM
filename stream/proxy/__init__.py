"""
Lakeshore Proxy Service

A lightweight HTTP proxy that routes vLLM inference requests to Lakeshore
via either Globus Compute or SSH port forwarding.

This proxy sits between LiteLLM and Lakeshore, allowing LiteLLM to:
- Maintain unified API gateway architecture
- Handle cost tracking and token counting
- Provide consistent logging and monitoring

The proxy transparently routes requests using the configured transport method.
"""
