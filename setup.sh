#!/bin/bash
# =============================================================================
# RKLLM API Server — Zero-Configuration Setup Script
# =============================================================================
# One-command installer for a fresh Armbian/Ubuntu system on RK3588 boards.
#
# What this script does:
#   1. Installs system packages (python3, venv, build-essential, git, git-lfs)
#   2. Checks RKNPU driver status
#   3. Installs RKLLM Runtime v1.2.3 (library + binary) if not already present
#   4. Creates a Python virtual environment with all dependencies
#   5. Creates ~/models directory for .rkllm model files
#   6. Applies NPU frequency fix for consistent performance
#   7. Installs a systemd service for auto-start on boot
#
# Tested on:
#   - Orange Pi 5 Plus (16 GB RAM, RK3588)
#   - Armbian Pelochus 24.11.0 (jammy vendor, RKNPU driver 0.9.8)
#     https://github.com/Pelochus/armbian-build-rknpu-updates/releases
#
# The script is IDEMPOTENT — safe to run multiple times.  It detects what
# is already installed (rkllm binary at /usr/bin or /usr/local/bin,
# librkllmrt.so at /lib or /usr/lib, existing systemd services) and
# skips those steps.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# The script will prompt for sudo when needed. Do NOT run the entire
# script as root — it creates user-level files (venv, models dir) that
# should be owned by your normal user account.
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these if your paths differ
# =============================================================================
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"    # Where this script lives (repo root)
MODELS_DIR="$HOME/models"                        # Where .rkllm models are stored
VENV_DIR="$INSTALL_DIR/venv"                     # Python virtual environment
RKNN_LLM_DIR="$HOME/rknn-llm"                   # rknn-llm repo clone location
SERVICE_NAME="rkllm-api"                         # systemd service name
BIND_ADDRESS="0.0.0.0"                           # Listen address
BIND_PORT="8000"                                 # Listen port
GUNICORN_TIMEOUT="300"                           # Gunicorn worker timeout (seconds)
RKLLM_LOG_LEVEL="1"                              # 0=silent, 1=stats, 2=debug

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

check_arch() {
    local arch
    arch=$(uname -m)
    if [[ "$arch" != "aarch64" ]]; then
        error "This script requires an aarch64 (ARM64) system. Detected: $arch"
        exit 1
    fi
    success "Architecture: $arch"
}

check_not_root() {
    if [[ "$EUID" -eq 0 ]]; then
        error "Do NOT run this script as root."
        echo "  The script will ask for sudo when needed."
        echo "  Run as your normal user: ./setup.sh"
        exit 1
    fi
    success "Running as user: $USER"
}

check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        info "Some steps require sudo. You may be prompted for your password."
        sudo -v || { error "sudo access required. Aborting."; exit 1; }
    fi
    success "sudo access confirmed"
}

separator() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# =============================================================================
# STEP 1: PRE-FLIGHT CHECKS
# =============================================================================

separator "RKLLM API Server — Zero-Configuration Setup"

# Track whether the service file was changed (used later for restart prompt)
SERVICE_CHANGED=false

echo "  Install directory : $INSTALL_DIR"
echo "  Models directory  : $MODELS_DIR"
echo "  Venv directory    : $VENV_DIR"
echo "  Service name      : $SERVICE_NAME"
echo "  Bind              : $BIND_ADDRESS:$BIND_PORT"
echo ""

check_arch
check_not_root
check_sudo

# =============================================================================
# STEP 2: SYSTEM PACKAGES
# =============================================================================

separator "Step 1/7 — Installing System Packages"

PACKAGES=(
    python3
    python3-venv
    python3-pip
    python3-dev
    build-essential
    git
    git-lfs
)

info "Updating package lists..."
sudo apt update -qq

info "Installing: ${PACKAGES[*]}"
sudo apt install -y -qq "${PACKAGES[@]}"

# Initialize git-lfs (needed for downloading models later)
git lfs install --skip-smudge 2>/dev/null || true

success "System packages installed"

# =============================================================================
# STEP 3: CHECK RKNPU DRIVER
# =============================================================================

separator "Step 2/7 — Checking RKNPU Driver"

DRIVER_VERSION=""

# Method 1: sysfs (may need root for debugfs)
if [[ -f /sys/kernel/debug/rknpu/version ]]; then
    DRIVER_VERSION=$(sudo cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo "")
fi

# Method 2: dmesg (may need sudo on newer kernels)
if [[ -z "$DRIVER_VERSION" ]]; then
    DRIVER_VERSION=$(sudo dmesg 2>/dev/null | grep -ioP 'rknpu.*version\s+\K[\d.]+' | tail -1 || echo "")
fi

# Method 3: Check if the device node exists (driver is loaded even if version unreadable)
if [[ -z "$DRIVER_VERSION" && -c /dev/rknpu ]]; then
    DRIVER_VERSION="present (version unreadable)"
fi

if [[ -n "$DRIVER_VERSION" ]]; then
    success "RKNPU driver detected: v$DRIVER_VERSION"
    if [[ "$DRIVER_VERSION" != "present"* && "$DRIVER_VERSION" < "0.9.6" ]]; then
        warn "Driver $DRIVER_VERSION is older than recommended (0.9.8)."
        warn "Consider using Pelochus Armbian builds which include 0.9.8:"
        warn "  https://github.com/Pelochus/armbian-build-rknpu-updates/releases"
    fi
else
    warn "Could not detect RKNPU driver version."
    warn "If using Pelochus Armbian 24.11.0, the driver (0.9.8) is already included."
    warn "Continuing anyway — the runtime will fail at model load time if the driver is missing."
fi

# =============================================================================
# STEP 4: RKLLM RUNTIME (librkllmrt.so + rkllm binary)
# =============================================================================

separator "Step 3/7 — Installing RKLLM Runtime v1.2.3"

# --- Detect existing rkllm binary (may be at /usr/bin or /usr/local/bin) ---
RKLLM_BIN=""
if command -v rkllm &>/dev/null; then
    RKLLM_BIN=$(which rkllm)
    success "rkllm binary already installed: $RKLLM_BIN"
    SKIP_BINARY=true
else
    # Search common locations
    for candidate in /usr/local/bin/rkllm /usr/bin/rkllm; do
        if [[ -x "$candidate" ]]; then
            RKLLM_BIN="$candidate"
            success "rkllm binary found at: $RKLLM_BIN (not in PATH — will fix)"
            # Add to PATH via profile if needed
            break
        fi
    done
    if [[ -z "$RKLLM_BIN" ]]; then
        SKIP_BINARY=false
        info "rkllm binary not found — will compile from source"
    else
        SKIP_BINARY=true
    fi
fi

# --- Detect existing runtime library (may be at /lib/ or /usr/lib/) ---
LIB_INSTALLED=false
if ldconfig -p 2>/dev/null | grep -q librkllmrt; then
    LIB_PATH=$(ldconfig -p 2>/dev/null | grep librkllmrt | awk '{print $NF}' | head -1)
    success "librkllmrt.so already installed: $LIB_PATH"
    LIB_INSTALLED=true
elif [[ -f /usr/lib/librkllmrt.so || -f /lib/librkllmrt.so ]]; then
    # Library exists but ldconfig doesn't know about it — fix that
    warn "librkllmrt.so found on disk but not in ldconfig — running ldconfig..."
    sudo ldconfig
    if ldconfig -p 2>/dev/null | grep -q librkllmrt; then
        success "librkllmrt.so now registered with ldconfig"
        LIB_INSTALLED=true
    fi
fi

# --- Install missing components ---
if [[ "$SKIP_BINARY" == false || "$LIB_INSTALLED" == false ]]; then

    # We need the rknn-llm repo for source files
    # Check common locations: $HOME/rknn-llm, $HOME/Downloads/ezrknpu/ezrknn-llm
    RKNN_FOUND=false
    for candidate_dir in "$RKNN_LLM_DIR" "$HOME/Downloads/ezrknpu/ezrknn-llm"; do
        if [[ -d "$candidate_dir/rkllm-runtime" ]]; then
            RKNN_LLM_DIR="$candidate_dir"
            RKNN_FOUND=true
            success "rknn-llm repo found at: $RKNN_LLM_DIR"
            break
        fi
    done

    if [[ "$RKNN_FOUND" == false ]]; then
        if [[ -d "$RKNN_LLM_DIR" ]]; then
            # Directory exists but missing expected structure — might be incomplete
            warn "rknn-llm directory exists at $RKNN_LLM_DIR but missing rkllm-runtime/"
            warn "Re-cloning..."
            rm -rf "$RKNN_LLM_DIR"
        fi
        info "Cloning rknn-llm repository..."
        git clone --depth 1 https://github.com/airockchip/rknn-llm.git "$RKNN_LLM_DIR"
        success "rknn-llm cloned to $RKNN_LLM_DIR"
    fi

    # Install shared library if missing
    if [[ "$LIB_INSTALLED" == false ]]; then
        LIB_SRC="$RKNN_LLM_DIR/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so"
        if [[ -f "$LIB_SRC" ]]; then
            info "Installing librkllmrt.so to /usr/lib/..."
            sudo cp "$LIB_SRC" /usr/lib/
            sudo ldconfig
            success "librkllmrt.so installed"
        else
            error "librkllmrt.so not found in rknn-llm repo at expected path:"
            error "  $LIB_SRC"
            exit 1
        fi
    fi

    # Compile llm_demo binary if missing
    if [[ "$SKIP_BINARY" == false ]]; then
        # Search for llm_demo.cpp in the rknn-llm repo
        DEMO_SRC=$(find "$RKNN_LLM_DIR" -name "llm_demo.cpp" -type f 2>/dev/null | head -1)

        if [[ -n "$DEMO_SRC" ]]; then
            DEMO_DIR=$(dirname "$DEMO_SRC")
            DEPLOY_DIR=$(dirname "$DEMO_DIR")  # Go up from src/ to deploy/
            info "Compiling rkllm (llm_demo) binary natively..."
            info "Source: $DEMO_SRC"

            cd "$DEPLOY_DIR"
            g++ -O2 -o rkllm "$DEMO_SRC" \
                -I"$RKNN_LLM_DIR/rkllm-runtime/Linux/librkllm_api/include" \
                -L"$RKNN_LLM_DIR/rkllm-runtime/Linux/librkllm_api/aarch64" \
                -lrkllmrt -lpthread

            sudo cp rkllm /usr/local/bin/
            RKLLM_BIN="/usr/local/bin/rkllm"
            cd "$INSTALL_DIR"

            success "rkllm binary compiled and installed to /usr/local/bin/rkllm"
        else
            error "llm_demo.cpp not found anywhere in $RKNN_LLM_DIR"
            echo ""
            echo "  This can happen if the rknn-llm repo structure changed or"
            echo "  the examples were removed.  You have two options:"
            echo ""
            echo "  Option 1: Re-clone the full repo (not shallow):"
            echo "    rm -rf $RKNN_LLM_DIR"
            echo "    git clone https://github.com/airockchip/rknn-llm.git $RKNN_LLM_DIR"
            echo "    Then re-run this script."
            echo ""
            echo "  Option 2: If you already have a compiled rkllm binary,"
            echo "    copy it to /usr/local/bin/:"
            echo "    sudo cp /path/to/rkllm /usr/local/bin/"
            echo "    Then re-run this script."
            echo ""
            exit 1
        fi
    fi
else
    success "RKLLM runtime already fully installed — skipping"
fi

# Verify
info "Verifying RKLLM runtime..."
RKLLM_BIN_FINAL=$(which rkllm 2>/dev/null || echo "")
RKLLM_LIB_FINAL=$(ldconfig -p 2>/dev/null | grep librkllmrt | awk '{print $NF}' | head -1)

if [[ -n "$RKLLM_LIB_FINAL" && -n "$RKLLM_BIN_FINAL" ]]; then
    success "Runtime verification passed"
    success "  Binary : $RKLLM_BIN_FINAL"
    success "  Library: $RKLLM_LIB_FINAL"
else
    if [[ -z "$RKLLM_LIB_FINAL" ]]; then
        error "librkllmrt.so not found in ldconfig"
    fi
    if [[ -z "$RKLLM_BIN_FINAL" ]]; then
        error "rkllm binary not found in PATH"
    fi
    error "Runtime verification failed. Check the output above."
    exit 1
fi

# =============================================================================
# STEP 5: PYTHON VIRTUAL ENVIRONMENT
# =============================================================================

separator "Step 4/7 — Setting Up Python Virtual Environment"

if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    success "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
fi

# Activate venv and install dependencies
info "Installing Python dependencies in venv..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet
pip install flask flask-cors gunicorn gevent --quiet

success "Python dependencies installed:"
pip list --format=columns | grep -iE "flask|gunicorn|gevent" | while read -r line; do
    echo "    $line"
done

deactivate

# =============================================================================
# STEP 6: MODELS DIRECTORY
# =============================================================================

separator "Step 5/7 — Creating Models Directory"

mkdir -p "$MODELS_DIR"
success "Models directory: $MODELS_DIR"

# Check if any models are already present
MODEL_COUNT=$(find "$MODELS_DIR" -name "*.rkllm" 2>/dev/null | wc -l)
if [[ "$MODEL_COUNT" -gt 0 ]]; then
    success "Found $MODEL_COUNT .rkllm model(s) in $MODELS_DIR"
    find "$MODELS_DIR" -name "*.rkllm" -exec basename {} \; | while read -r m; do
        echo "    - $m"
    done
else
    warn "No .rkllm models found in $MODELS_DIR"
    echo ""
    echo "  Download models from HuggingFace:"
    echo "    https://huggingface.co/GatekeeperZA"
    echo ""
    echo "  Quick download (Qwen3-1.7B — recommended):"
    echo "    mkdir -p ~/models/Qwen3-1.7B-4K"
    echo "    cd ~/models/Qwen3-1.7B-4K"
    echo "    git lfs install"
    echo "    git clone https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3 ."
    echo ""
fi

# =============================================================================
# STEP 7: NPU FREQUENCY FIX
# =============================================================================

separator "Step 6/7 — NPU Frequency Fix"

# Create a persistent frequency fix script (the actual commands)
NPU_FIX_SCRIPT="/usr/local/bin/rk3588-npu-fix-freq.sh"

if [[ -x "$NPU_FIX_SCRIPT" ]]; then
    success "Frequency fix script already exists: $NPU_FIX_SCRIPT"
else
    info "Installing NPU frequency fix script..."
    sudo tee "$NPU_FIX_SCRIPT" > /dev/null << 'FREQSCRIPT'
#!/bin/bash
# RK3588 NPU + CPU frequency governor fix for consistent inference performance.
# Sets performance governor to prevent clock scaling during inference.

# NPU frequency (if available)
NPU_GOV="/sys/class/devfreq/fdab0000.npu/governor"
if [[ -f "$NPU_GOV" ]]; then
    echo performance > "$NPU_GOV" 2>/dev/null && echo "[OK] NPU governor: performance" || echo "[SKIP] NPU governor"
fi

# CPU governors (all policy groups)
for gov in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    if [[ -f "$gov" ]]; then
        echo performance > "$gov" 2>/dev/null
    fi
done
echo "[OK] CPU governors: performance"

# DMC (memory controller) if available
DMC_GOV="/sys/class/devfreq/dmc/governor"
if [[ -f "$DMC_GOV" ]]; then
    echo performance > "$DMC_GOV" 2>/dev/null && echo "[OK] DMC governor: performance" || echo "[SKIP] DMC governor"
fi
FREQSCRIPT
    sudo chmod +x "$NPU_FIX_SCRIPT"
    success "Frequency fix script installed: $NPU_FIX_SCRIPT"
fi

# Apply it now
info "Applying frequency fix..."
sudo "$NPU_FIX_SCRIPT" || warn "Some frequency settings may have failed (non-critical)"

# Persistence: prefer systemd service over rc.local
# The systemd service is created in Step 7 below (fix-freq.service).
# If an old rc.local entry exists, it's harmless (runs before systemd service).

# =============================================================================
# STEP 8: SYSTEMD SERVICE
# =============================================================================

separator "Step 7/7 — Creating Systemd Service"

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
FREQ_SERVICE_FILE="/etc/systemd/system/fix-freq.service"

# --- Frequency fix as a proper systemd service (replaces rc.local method) ---
if [[ -f "$FREQ_SERVICE_FILE" ]]; then
    success "Frequency fix service already exists: $FREQ_SERVICE_FILE"
else
    info "Creating frequency fix systemd service..."
    sudo tee "$FREQ_SERVICE_FILE" > /dev/null << 'EOF'
[Unit]
Description=Fix NPU/CPU/DDR frequencies for RKLLM performance
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/rk3588-npu-fix-freq.sh

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable fix-freq.service
    sudo systemctl start fix-freq.service 2>/dev/null || true
    success "Frequency fix service created and enabled"
fi

# --- Main API service ---
# Detect the rkllm binary location for the service PATH
RKLLM_BIN_DIR=$(dirname "$(which rkllm 2>/dev/null || echo /usr/local/bin/rkllm)")

EXISTING_SERVICE=false
SERVICE_CHANGED=false
if [[ -f "$SERVICE_FILE" ]]; then
    EXISTING_SERVICE=true
    info "Existing service found: $SERVICE_FILE"

    # Check if it points to our install directory
    EXISTING_WORKDIR=$(grep -oP 'WorkingDirectory=\K.*' "$SERVICE_FILE" 2>/dev/null || echo "")
    if [[ "$EXISTING_WORKDIR" == "$INSTALL_DIR" ]]; then
        # Check if venv path matches (upgrade from non-venv to venv)
        if grep -q "$VENV_DIR/bin/gunicorn" "$SERVICE_FILE" 2>/dev/null; then
            success "Service already configured correctly for $INSTALL_DIR"
        else
            info "Service exists but needs updating (venv path changed)"
            SERVICE_CHANGED=true
        fi
    else
        warn "Existing service points to: $EXISTING_WORKDIR"
        warn "This script's install dir is: $INSTALL_DIR"
        echo ""
        read -rp "$(echo -e "${YELLOW}Update service to use this directory? [Y/n]: ${NC}")" UPDATE_SVC
        UPDATE_SVC=${UPDATE_SVC:-Y}
        if [[ "$UPDATE_SVC" =~ ^[Yy] ]]; then
            SERVICE_CHANGED=true
        else
            info "Keeping existing service configuration"
        fi
    fi
fi

if [[ "$EXISTING_SERVICE" == false || "$SERVICE_CHANGED" == true ]]; then
    # Stop existing service before overwriting
    if [[ "$EXISTING_SERVICE" == true ]]; then
        info "Stopping existing service..."
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    fi

    info "Creating systemd service: $SERVICE_NAME"
    sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=RKLLM API Server (OpenAI-compatible, RK3588 NPU)
After=network.target fix-freq.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$(id -gn)
WorkingDirectory=$INSTALL_DIR

# Environment
Environment="PATH=$VENV_DIR/bin:$RKLLM_BIN_DIR:/usr/local/bin:/usr/bin:/bin"
Environment="RKLLM_LOG_LEVEL=$RKLLM_LOG_LEVEL"
Environment="RKLLM_API_LOG_LEVEL=INFO"

# Run with gunicorn (single worker — NPU loads one model at a time)
ExecStart=$VENV_DIR/bin/gunicorn \\
    -w 1 \\
    -k gevent \\
    --timeout $GUNICORN_TIMEOUT \\
    -b ${BIND_ADDRESS}:${BIND_PORT} \\
    --access-logfile - \\
    --error-logfile - \\
    api:app

# Restart policy
Restart=on-failure
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=3

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security hardening (comment out if causing issues)
ProtectHome=no
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=$INSTALL_DIR $MODELS_DIR /tmp

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    success "Service created and enabled: $SERVICE_NAME"
else
    success "Service unchanged"
fi
echo ""
echo "  Start now:     sudo systemctl start $SERVICE_NAME"
echo "  Stop:          sudo systemctl stop $SERVICE_NAME"
echo "  Status:        sudo systemctl status $SERVICE_NAME"
echo "  Logs:          journalctl -u $SERVICE_NAME -f"
echo "  Disable:       sudo systemctl disable $SERVICE_NAME"

# =============================================================================
# OPTIONAL: START THE SERVICE NOW?
# =============================================================================

echo ""
if [[ "$MODEL_COUNT" -gt 0 ]]; then
    # Check if already running
    if sudo systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        success "API server is already running!"
        if [[ "$SERVICE_CHANGED" == true ]]; then
            read -rp "$(echo -e "${YELLOW}Service was updated. Restart now? [Y/n]: ${NC}")" RESTART_NOW
            RESTART_NOW=${RESTART_NOW:-Y}
            if [[ "$RESTART_NOW" =~ ^[Yy] ]]; then
                sudo systemctl restart "$SERVICE_NAME"
                sleep 2
                if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
                    success "API server restarted successfully!"
                else
                    warn "Restart may have failed. Check: journalctl -u $SERVICE_NAME -f"
                fi
            fi
        fi
    else
        read -rp "$(echo -e "${GREEN}Models found. Start the API server now? [Y/n]: ${NC}")" START_NOW
        START_NOW=${START_NOW:-Y}
        if [[ "$START_NOW" =~ ^[Yy] ]]; then
            sudo systemctl start "$SERVICE_NAME"
            sleep 2
            if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
                success "API server is running!"
                echo ""
                echo "  API endpoint: http://$(hostname -I | awk '{print $1}'):$BIND_PORT/v1/models"
                echo "  Test:         curl http://localhost:$BIND_PORT/v1/models"
            else
                warn "Service started but may not be ready yet. Check:"
                echo "  sudo systemctl status $SERVICE_NAME"
                echo "  journalctl -u $SERVICE_NAME -f"
            fi
        fi
    fi
else
    info "No models found — service enabled but not started."
    echo "  Download a model first, then: sudo systemctl start $SERVICE_NAME"
fi

# =============================================================================
# SUMMARY
# =============================================================================

separator "Setup Complete!"

echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  RKLLM API Server — Installation Summary               │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │                                                         │"
echo "  │  API Server   : $INSTALL_DIR/api.py"
echo "  │  Venv         : $VENV_DIR"
echo "  │  Models       : $MODELS_DIR ($MODEL_COUNT model(s))"
echo "  │  Service      : $SERVICE_NAME (systemd)"
echo "  │  Endpoint     : http://0.0.0.0:$BIND_PORT/v1"
echo "  │  rkllm binary : ${RKLLM_BIN_FINAL:-$(which rkllm 2>/dev/null || echo 'not found')}"
echo "  │  Runtime lib  : ${RKLLM_LIB_FINAL:-$(ldconfig -p 2>/dev/null | grep librkllmrt | awk '{print $NF}' | head -1)}"
echo "  │                                                         │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │  NEXT STEPS:                                            │"
echo "  │                                                         │"
if [[ "$MODEL_COUNT" -eq 0 ]]; then
echo "  │  1. Download a model:                                   │"
echo "  │     mkdir -p ~/models/Qwen3-1.7B-4K                    │"
echo "  │     cd ~/models/Qwen3-1.7B-4K                          │"
echo "  │     git clone https://huggingface.co/GatekeeperZA/\\    │"
echo "  │       Qwen3-1.7B-RKLLM-v1.2.3 .                       │"
echo "  │                                                         │"
echo "  │  2. Start the server:                                   │"
echo "  │     sudo systemctl start $SERVICE_NAME                     │"
echo "  │                                                         │"
echo "  │  3. Connect Open WebUI:                                 │"
echo "  │     Admin > Settings > Connections                      │"
echo "  │     API URL: http://<this-ip>:$BIND_PORT/v1                 │"
else
echo "  │  1. Connect Open WebUI:                                 │"
echo "  │     Admin > Settings > Connections                      │"
echo "  │     API URL: http://<this-ip>:$BIND_PORT/v1                 │"
fi
echo "  │                                                         │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
echo "  Manual start (without systemd):"
echo "    source $VENV_DIR/bin/activate"
echo "    gunicorn -w 1 -k gevent --timeout 300 -b 0.0.0.0:$BIND_PORT api:app"
echo ""
