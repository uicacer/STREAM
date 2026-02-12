#!/usr/bin/env python3
"""
Lakeshore Authentication Setup
==============================

Run this ONCE to authenticate with Globus Compute for Lakeshore access.
After running, credentials are saved and you won't need to do this again
(until they expire, typically after several months).

USAGE:
    python scripts/setup_lakeshore.py

This is similar to how other CLI tools work:
    - aws configure
    - gcloud auth login
    - gh auth login
    - docker login

After authentication, the React/Streamlit frontends will automatically
detect the credentials and enable the Lakeshore tier.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    print("=" * 60)
    print("🏫 STREAM - Lakeshore Setup")
    print("=" * 60)
    print()
    print("This will authenticate you with Globus Compute to access")
    print("the UIC Lakeshore HPC cluster.")
    print()
    print("A browser window will open for you to log in with your")
    print("university credentials.")
    print()
    print("-" * 60)

    try:
        from stream.middleware.core.globus_auth import (
            authenticate_with_browser_callback,
            is_authenticated,
        )

        # Check if already authenticated
        if is_authenticated():
            print("✅ Already authenticated with Globus Compute!")
            print()
            print("You're all set. Lakeshore tier is available.")
            print("=" * 60)
            return

        # Authenticate
        print("Opening browser for Globus login...")
        print()

        success, message = authenticate_with_browser_callback()

        print()
        if success:
            print("✅ " + message)
            print()
            print("Lakeshore tier is now available!")
            print("You can close this terminal and use the STREAM app.")
        else:
            print("❌ " + message)
            print()
            print("Please try again or check your credentials.")

    except ImportError as e:
        print(f"❌ Error: Could not import Globus modules: {e}")
        print()
        print("Make sure you have the Globus SDK installed:")
        print("    pip install globus-compute-sdk")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
