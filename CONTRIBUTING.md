# Contributing to RKLLM API Server

Thank you for your interest in contributing! This project is an OpenAI-compatible API server for Rockchip NPU (RK3588/RK3576) hardware. Contributions that improve compatibility, performance, or documentation are welcome.

## Before You Start

- Check [open issues](https://github.com/GatekeeperZA/RKLLM-API-Server/issues) to avoid duplicating work
- For large changes, open an issue first to discuss the approach
- This project targets **Orange Pi 5 Plus / RK3588** with RKLLM runtime v1.2.3 — changes that break this hardware target will not be merged

## Hardware Requirements for Testing

To test changes against real hardware you need:
- Orange Pi 5 / 5 Plus or similar RK3588 board
- RKLLM runtime (`librkllmrt.so` v1.2.3)
- At least one `.rkllm` model file in `~/models/`

If you don't have the hardware, clearly state this in your PR — we can run tests on our end.

## Development Setup

```bash
git clone https://github.com/GatekeeperZA/RKLLM-API-Server.git
cd RKLLM-API-Server
python3 -m venv .venv
source .venv/bin/activate
pip install flask flask-cors gunicorn numpy Pillow prometheus-flask-exporter prometheus-client
```

Run the test suite (requires live server):
```bash
python tests/e2e_test.py                  # End-to-end API tests
python tests/test_openwebui.py            # Open WebUI integration tests
```

## Pull Request Guidelines

1. **One concern per PR** — fix one bug or add one feature
2. **Test on hardware if possible** — include test results in the PR description
3. **Don't break existing behaviour** — all endpoints must remain OpenAI-compatible
4. **No new dependencies without discussion** — the server is intentionally lean
5. **Match the existing code style** — snake_case, type hints where used, minimal comments

## What We're Looking For

- Bug fixes with reproduction steps
- New model family support (sampling profiles, capability detection)
- Performance improvements to the RKLLM ctypes binding
- Documentation improvements and clarifications
- Additional test coverage

## What We Won't Merge

- Changes that add cloud API dependencies to `api.py` (the local API must remain self-contained)
- New endpoints that duplicate existing OpenAI API functionality differently
- Large refactors without prior discussion
- Changes that break the Dockerfile or `setup.sh` automated install

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:
- Hardware (board, RAM, NPU driver version)
- RKLLM runtime version (`~/librkllmrt.so` or `/usr/lib/librkllmrt.so`)
- Model name and quantisation
- Relevant section of `api.log`

## License

By contributing, you agree that your contributions will be licensed under the same terms as this project (see [LICENSE](LICENSE)).
