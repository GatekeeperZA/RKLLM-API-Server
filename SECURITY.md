# Security Policy

## Supported Versions

Only the latest commit on `main` is actively maintained. This is a single-device local inference server — there are no versioned releases with backported security fixes.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| Older commits | ❌ |

## Scope

This server is designed to run on a **local network** (Orange Pi 5 / RK3588) and is not intended to be exposed directly to the internet. The primary attack surface is:

- The REST API on port 8000 (no authentication by default)
- The embed API on port 8001 (no authentication by default)
- The Hermes Agent gateway on port 8642 (API key required)

If you are exposing this server beyond your local network, you should place it behind a reverse proxy with authentication (e.g. Nginx + basic auth, or a VPN).

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public issue**.

Report it privately by emailing: **admin@theroot.za.net**

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix if you have one

You can expect an acknowledgement within 48 hours and a fix or mitigation within 14 days depending on severity.

Confirmed vulnerabilities will be disclosed publicly in the [Known Issues & Fixes](README.md#known-issues--fixes) section of the README once a fix is available.

## Known Security Assumptions

- **No auth on port 8000/8001 by default** — intended for trusted local network use only
- **`--privileged` Docker flag** — required for NPU device access; do not run untrusted workloads in the same container
- **Model files are trusted** — `.rkllm` files are executed by the Rockchip NPU runtime; only use models from trusted sources
