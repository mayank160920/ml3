"""
Opens ngrok tunnels for already-running FastAPI and Streamlit services.

Run AFTER starting FastAPI and Streamlit manually:

  Terminal 1: uvicorn api.main:app --host 0.0.0.0 --port 8000
  Terminal 2: streamlit run streamlit_app/app.py --server.port 8501
  Terminal 3: python scripts/ngrok_only.py --ngrok_token YOUR_TOKEN

Get your token: https://dashboard.ngrok.com/get-started/your-authtoken
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

# Load .env
ROOT = Path(__file__).resolve().parent.parent
env_file = ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open ngrok tunnels for CMSVS")
    p.add_argument("--ngrok_token", required=True)
    p.add_argument("--api_port",       type=int, default=8000)
    p.add_argument("--streamlit_port", type=int, default=8501)
    return p.parse_args()


def is_port_open(port: int) -> bool:
    """Check if something is already listening on this port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: int = 60, label: str = "") -> bool:
    print(f"  Waiting for {label} on port {port}...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open(port):
            print(" ✅")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(" ❌")
    return False


def main() -> None:
    args = parse_args()

    # Validate token
    bad = {"YOUR_TOKEN", "YOUR_REAL_TOKEN_HERE", ""}
    if args.ngrok_token.upper() in {t.upper() for t in bad}:
        print("❌ Please provide your real ngrok token.")
        print("   Get it at: https://dashboard.ngrok.com/get-started/your-authtoken")
        sys.exit(1)

    print(f"\n[ngrok] Token: {args.ngrok_token[:12]}...")

    # Check services are running
    print("\n[Check] Verifying services are running...")

    if not is_port_open(args.api_port):
        print(f"\n⚠️  FastAPI is NOT running on port {args.api_port}.")
        print(f"   Start it first:")
        print(f"   uvicorn api.main:app --host 0.0.0.0 --port {args.api_port}")
        print(f"\n   Waiting up to 60s for FastAPI to start...")
        if not wait_for_port(args.api_port, timeout=60, label="FastAPI"):
            print("❌ FastAPI never started. Exiting.")
            sys.exit(1)
    else:
        print(f"  FastAPI  port {args.api_port}: ✅ running")

    if not is_port_open(args.streamlit_port):
        print(f"\n⚠️  Streamlit is NOT running on port {args.streamlit_port}.")
        print(f"   Start it first:")
        print(f"   streamlit run streamlit_app/app.py --server.port {args.streamlit_port}")
        print(f"\n   Waiting up to 60s for Streamlit to start...")
        if not wait_for_port(args.streamlit_port, timeout=60, label="Streamlit"):
            print("❌ Streamlit never started. Exiting.")
            sys.exit(1)
    else:
        print(f"  Streamlit port {args.streamlit_port}: ✅ running")

    # Configure ngrok
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = args.ngrok_token

    # Open FastAPI tunnel
    print(f"\n[ngrok] Opening FastAPI tunnel (port {args.api_port})...")
    try:
        api_tunnel = ngrok.connect(args.api_port, "http")
        api_url = api_tunnel.public_url
        print(f"[ngrok] ✅ FastAPI → {api_url}")
    except Exception as exc:
        print(f"❌ FastAPI tunnel failed: {exc}")
        sys.exit(1)

    # Open Streamlit tunnel
    print(f"[ngrok] Opening Streamlit tunnel (port {args.streamlit_port})...")
    try:
        st_tunnel = ngrok.connect(args.streamlit_port, "http")
        st_url = st_tunnel.public_url
        print(f"[ngrok] ✅ Streamlit → {st_url}")
    except Exception as exc:
        print(f"❌ Streamlit tunnel failed: {exc}")
        sys.exit(1)

    # Print URLs
    w = 58
    print(f"""
╔{"═" * w}╗
║{"  🚀  CMSVS is now publicly accessible!  ":^{w}}║
╠{"═" * w}╣
║{"":^{w}}║
║  📊  Streamlit App — share this URL with users:          ║
║  {st_url:<{w-2}} ║
║{"":^{w}}║
║  🔌  FastAPI Docs:                                       ║
║  {(api_url + "/docs"):<{w-2}} ║
║{"":^{w}}║
║  ⚠️   NOTE: Update Streamlit to use the FastAPI URL:     ║
║  Set CMSVS_API_URL={api_url:<{w-22}} ║
║  in the terminal where Streamlit is running.             ║
║{"":^{w}}║
║  Press Ctrl+C to close tunnels.                          ║
╚{"═" * w}╝
""")

    # Keep tunnels open
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nClosing tunnels...")
        ngrok.kill()
        print("✅ Tunnels closed.")


if __name__ == "__main__":
    main()