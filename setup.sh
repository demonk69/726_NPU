#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[MISS]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC}  $*"; }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }

INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=1 ;;
        --help|-h) echo "Usage: $0 [--install]" ; echo "" ; echo "  (no flag)  Check prerequisites only, print missing items." ; echo "  --install  Auto-install missing packages/tools/files." ; exit 0 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

SUDO=""
if command -v sudo &>/dev/null; then
    SUDO="sudo"
fi

PASS=0
MISS=0

# ── helpers ──

install_apt() {
    local pkg="$1"
    if dpkg -l "$pkg" &>/dev/null; then return 0; fi
    info "installing $pkg ..."
    $SUDO apt-get update -qq 2>/dev/null
    $SUDO apt-get install -y -qq "$pkg"
}

# ── python3 ──

echo ""
echo "=== Python ==="

if command -v python3 &>/dev/null; then
    ok "python3 $(python3 --version 2>&1 | awk '{print $2}')"
    PASS=$((PASS + 1))
else
    warn "python3"
    if [[ $INSTALL -eq 1 ]]; then
        install_apt python3 && install_apt python3-pip
        if command -v python3 &>/dev/null; then ok "python3 installed"; ((PASS++))
        else err "python3 install failed"; fi
    fi
    MISS=$((MISS + 1))
fi

# ── PyTorch CPU ──

if python3 -c "import torch" 2>/dev/null; then
    ok "PyTorch  $(python3 -c 'import torch; print(torch.__version__)')"
    PASS=$((PASS + 1))
else
    warn "PyTorch CPU"
    if [[ $INSTALL -eq 1 ]]; then
        python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
        if python3 -c "import torch" 2>/dev/null; then ok "PyTorch CPU installed"; ((PASS++))
        else err "PyTorch install failed"; fi
    fi
    MISS=$((MISS + 1))
fi

# ── Pillow ──

if python3 -c "import PIL" 2>/dev/null; then
    ok "Pillow  $(python3 -c 'import PIL; print(PIL.__version__)')"
    PASS=$((PASS + 1))
else
    warn "Pillow"
    if [[ $INSTALL -eq 1 ]]; then
        python3 -m pip install --user Pillow
        if python3 -c "import PIL" 2>/dev/null; then ok "Pillow installed"; ((PASS++))
        else err "Pillow install failed"; fi
    fi
    MISS=$((MISS + 1))
fi

# ── Icarus Verilog ──

echo ""
echo "=== RTL Tools ==="

if command -v iverilog &>/dev/null && command -v vvp &>/dev/null; then
    ok "Icarus Verilog (iverilog / vvp)"
    PASS=$((PASS + 1))
else
    warn "Icarus Verilog"
    if [[ $INSTALL -eq 1 ]]; then
        install_apt iverilog
        if command -v iverilog &>/dev/null; then ok "Icarus installed"; ((PASS++))
        else err "Icarus install failed"; fi
    fi
    MISS=$((MISS + 1))
fi

# ── Verilator ──

VERILATOR_TAG="v5.030"

if command -v verilator &>/dev/null; then
    ver_ver=$(verilator --version 2>&1 | head -1)
    ok "Verilator  $ver_ver"
    PASS=$((PASS + 1))
else
    warn "Verilator"
    if [[ $INSTALL -eq 1 ]]; then
        info "building Verilator $VERILATOR_TAG from source ..."
        install_apt git
        install_apt make
        install_apt g++
        install_apt perl
        install_apt python3
        install_apt libfl-dev
        install_apt zlib1g-dev
        install_apt autoconf

        TMP=$(mktemp -d)
        git clone --depth 1 --branch "$VERILATOR_TAG" \
            https://github.com/verilator/verilator "$TMP/verilator"

        (
            cd "$TMP/verilator"
            autoconf
            ./configure
            make -j"$(nproc)"
            $SUDO make install
        )

        rm -rf "$TMP"

        if command -v verilator &>/dev/null; then
            ok "Verilator  $(verilator --version 2>&1 | head -1)"
            PASS=$((PASS + 1))
        else
            err "Verilator build failed"
        fi
    fi
    MISS=$((MISS + 1))
fi

# ── model checkpoint ──

echo ""
echo "=== Model Files ==="

PTH_DIR="$ROOT/RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat"
PTH_FILE="$PTH_DIR/qat_int8_quantized.pth"
PTH_URL="https://github.com/demonk69/726_NPU/releases/download/model-v4.1.0/qat_int8_quantized.pth"

if [[ -f "$PTH_FILE" ]]; then
    ok "checkpoint  qat_int8_quantized.pth ($(du -h "$PTH_FILE" | cut -f1))"
    PASS=$((PASS + 1))
else
    warn "checkpoint  $PTH_FILE"
    if [[ $INSTALL -eq 1 ]]; then
        mkdir -p "$PTH_DIR"

        if command -v curl &>/dev/null; then
            DL="curl -L -o"
        elif command -v wget &>/dev/null; then
            DL="wget -O"
        else
            install_apt curl
            DL="curl -L -o"
        fi

        $DL "$PTH_FILE" "$PTH_URL"
        if [[ -f "$PTH_FILE" ]]; then
            ok "checkpoint downloaded"
            PASS=$((PASS + 1))
            ((MISS--))
        else
            err "checkpoint download failed"
        fi
    fi
    MISS=$((MISS + 1))
fi

# ── model_plan.json ──

PLAN_FILE="$ROOT/sim/pth_repopt_probe/model_plan.json"

if [[ -f "$PLAN_FILE" ]]; then
    ok "model_plan  $PLAN_FILE"
    PASS=$((PASS + 1))
else
    if [[ -f "$PTH_FILE" ]]; then
        warn "model_plan  (checkpoint present, needs generation)"
        if [[ $INSTALL -eq 1 ]]; then
            python3 "$ROOT/tools/pth/pth_to_npu_assets.py" \
                --pth "$PTH_FILE" \
                --spec "$ROOT/tools/pth/examples/repopt_vgg_int8_spec.json" \
                --out-dir "$ROOT/sim/pth_repopt_probe"
            if [[ -f "$PLAN_FILE" ]]; then
                ok "model_plan.json generated"
                PASS=$((PASS + 1))
                ((MISS--))
            else
                err "model_plan generation failed"
            fi
        fi
        MISS=$((MISS + 1))
    else
        warn "model_plan  (needs checkpoint first)"
        MISS=$((MISS + 1))
    fi
fi

# ── summary ──

echo ""
echo "=============================================="
if [[ $MISS -eq 0 ]]; then
    echo -e "${GREEN}All $PASS prerequisites passed.${NC}"
else
    echo -e "${GREEN}PASS: $PASS${NC}  ${RED}MISS: $MISS${NC}"
    if [[ $INSTALL -eq 0 ]]; then
        echo ""
        echo "Run with --install to auto-install missing items:"
        echo -e "  ${CYAN}bash setup.sh --install${NC}"
    fi
fi
echo "=============================================="
