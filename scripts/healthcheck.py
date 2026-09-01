#!/usr/bin/env python3
"""
healthcheck.py — Used by Docker HEALTHCHECK and the CI smoke-test job.
Exit 0 = healthy, Exit 1 = unhealthy.
"""
import sys

def check():
    try:
        from config.settings import Settings
        Settings.validate()
        from langfuse import get_client
        _ = get_client()
        print("[healthcheck] OK — all settings valid")
        sys.exit(0)
    except ValueError as e:
        print(f"[healthcheck] FAIL — missing config: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[healthcheck] FAIL — unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    check()
