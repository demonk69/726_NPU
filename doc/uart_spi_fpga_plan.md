# UART/SPI FPGA Deployment Plan

Updated: 2026-05-26

Target direction: Xilinx FPGA board, SPI Flash for static assets, UART for image input and class output.

## Goal

Run the closed-loop VGG flow on FPGA without relying on testbench `$readmemh` initialization.

The upper PC performs image preprocessing and sends one quantized image per inference. FPGA returns one class byte.

## Target Runtime Flow

1. FPGA powers on.
2. Boot ROM runs from BRAM.
3. Boot ROM copies firmware from SPI Flash to SRAM.
4. Boot ROM copies static model assets from SPI Flash to DRAM.
5. Boot ROM transfers execution to the main firmware.
6. Firmware waits for 3072 image bytes over UART.
7. Firmware writes image bytes into the padded `ACT_A` buffer.
8. Firmware runs closed-loop inference.
9. Firmware sends one class byte over UART.
10. Firmware loops back and waits for the next image.

## PC-to-FPGA Protocol

Physical link: UART 8N1.

Initial baud target: 115200.

Per inference payload:

| Direction | Bytes | Meaning |
|---|---:|---|
| PC -> FPGA | 3072 | 3x32x32 signed INT8 image in CHW order |
| FPGA -> PC | 1 | Argmax class id, 0-9 |

At 115200 baud, 3072 bytes take about 0.27 seconds on the wire. This is negligible compared with current Verilator runtime and acceptable for first FPGA bring-up.

## Host Preprocessing

The host script should not require PyTorch for runtime inference.

`tools/host/preprocess.py` should:

1. Load JPEG/PNG with Pillow.
2. Convert to RGB.
3. Resize to 32x32 with bilinear interpolation.
4. Normalize with CIFAR-10 constants.
5. Quantize using fixed `input_scale` and `input_zero_point` extracted once from the checkpoint.
6. Emit 3072 signed INT8 bytes in CHW order.

The host client `tools/host/infer.py` should open the serial port, send the bytes, wait for one result byte, and print the class name.

## UART Peripheral Plan

Memory map proposal: `UART_BASE = 0x03000000`.

| Offset | Name | Access | Bits |
|---:|---|---|---|
| `0x00` | `UART_STATUS` | R | bit0=`RX_READY`, bit1=`TX_BUSY` |
| `0x04` | `UART_DATA` | R/W | read RX byte, write TX byte |
| `0x08` | `UART_DIV` | R/W | baud-rate divider |

Firmware behavior:

- Poll `RX_READY` before reading each byte.
- Poll `TX_BUSY == 0` before writing the result byte.
- No interrupt support is required for first bring-up.

## SPI Flash Plan

First implementation can use a read-only SPI Flash controller.

Memory map proposal: `SPI_BASE = 0x03000010`.

| Offset | Name | Access | Bits |
|---:|---|---|---|
| `0x00` | `SPI_CMD` | W | start read from 24-bit flash address |
| `0x04` | `SPI_STATUS` | R | bit0=`BUSY`, bit1=`DATA_VALID` |
| `0x08` | `SPI_DATA` | R | returned byte |

The boot ROM can poll the controller and copy byte or word streams into SRAM/DRAM.

## Flash Image Layout

Proposed `flash.bin` layout:

| Flash offset | Contents |
|---:|---|
| `0x000000` | Main firmware binary |
| `0x001000` | `fw_size` as 32-bit little-endian |
| `0x001004` | `dram_size` as 32-bit little-endian |
| `0x001008` | `dram_addr`, expected `0x00070000` |
| `0x00100C` | Static DRAM asset blob |

The static asset blob contains weights, bias, per-channel Q24 multipliers, classifier weights/biases, and layer descriptors. It does not contain the per-image activation input.

## RTL Work Items

| Item | File |
|---|---|
| UART peripheral | `rtl/soc/uart.v` |
| SPI Flash reader | `rtl/soc/spi_flash_reader.v` |
| Boot ROM | `rtl/soc/boot_rom.v` |
| SoC address decode and boot mux | `rtl/soc/soc_top.v` |
| UART testbench driver | `tb/` or integrated in closed-loop TB |
| SPI Flash behavioral model | `tb/spi_flash_model.v` |

## Host Tool Work Items

| Item | File |
|---|---|
| Extract model input quant constants | `tools/host/extract_input_quant.py` |
| Runtime image preprocessing | `tools/host/preprocess.py` |
| Serial inference client | `tools/host/infer.py` |
| Flash image packer | `tools/host/flash_gen.py` |

## Bring-Up Order

1. Implement and simulate UART with a tiny echo firmware.
2. Add UART image receive/send-result to closed-loop firmware, still using testbench-initialized DRAM.
3. Implement SPI Flash model and flash image packer.
4. Add Boot ROM and copy static assets from SPI Flash in simulation.
5. Port to Xilinx board constraints.
6. Program Flash and verify UART inference on board.

## Current Status

Not implemented yet. This document is the implementation plan for the next hardware-deployment stage.
