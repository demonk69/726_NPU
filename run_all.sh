#!/usr/bin/env bash
# Run VGG NPU tests
# Usage:
#   ./run_all.sh standard [img_idx]     # CIFAR-10 image index (default 0)
#   ./run_all.sh image <file.jpg>       # arbitrary image
#   ./run_all.sh closed_loop [args...]  # runtime closed-loop path; forwards args
#   ./run_all.sh im2col27               # im2col K=27 test
#   ./run_all.sh im2col576              # im2col K=576 test
#   ./run_all.sh regress                # full pipeline regression
#   ./run_all.sh all                    # run fast regression set
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RTL_SRC="$ROOT/sim/picorv32.v $ROOT/rtl/pe/*.v $ROOT/rtl/common/*.v $ROOT/rtl/buf/*.v $ROOT/rtl/array/*.v $ROOT/rtl/axi/*.v $ROOT/rtl/ctrl/*.v $ROOT/rtl/power/*.v $ROOT/rtl/soc/*.v $ROOT/rtl/top/*.v $ROOT/tb/tb_soc_vgg_e2e.v"

WARN_FLAGS="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSED -Wno-UNDRIVEN -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-CASEINCOMPLETE -Wno-COMBDLY -Wno-INITIALDLY -Wno-LITENDIAN"

test_standard() {
    echo "=== Standard E2E (1024 tiles, full pipeline) ==="
    "$ROOT/run_vgg_e2e.sh" "$@"
}

test_image() {
    echo "=== Classify: ${1:?missing image path} ==="
    "$ROOT/run_vgg_e2e.sh" --image "$1"
}

test_closed_loop() {
    echo "=== Runtime Closed Loop VGG ==="
    "$ROOT/run_vgg_closed_loop.sh" "$@"
}

test_im2col27() {
    echo "=== Im2col K=27 (stage1_0, 1 tile, NCHW input) ==="
    cd /tmp/nchw_verify && ./obj_dir/Vtb_soc_vgg_e2e 2>&1 | grep -E 'PASS|Cycles'
}

test_im2col576() {
    echo "=== Im2col K=576 (stage2_0, 1 tile, NCHW input) ==="
    cd /tmp/k576_test && ./obj_dir/Vtb_soc_vgg_e2e 2>&1 | grep -E 'PASS|Cycles'
}

test_regress() {
    echo "=== Full Pipeline Regression (1024 tiles, standalone FW) ==="
    verilator --binary --timing \
      -I/tmp/mp_full_fixed -Mdir /tmp/mp_full_fixed/obj_dir --top-module tb_soc_vgg_e2e \
      $WARN_FLAGS $RTL_SRC 2>&1 | tail -1
    timeout 1200 /tmp/mp_full_fixed/obj_dir/Vtb_soc_vgg_e2e 2>&1 | grep -E 'PASS|Cycles|TIMEOUT'
}

case "${1:-standard}" in
    standard) shift; test_standard "$@" ;;
    image)    shift; test_image "$@" ;;
    closed_loop|closed-loop|full) shift; test_closed_loop "$@" ;;
    im2col27) test_im2col27 ;;
    im2col576) test_im2col576 ;;
    regress) test_regress ;;
    all)
        test_im2col27
        test_im2col576
        test_standard 0
        test_regress
        ;;
    *)
        echo "Usage: $0 [standard [idx] | image <file> | closed_loop [args...] | im2col27 | im2col576 | regress | all]"
        exit 1
        ;;
esac
