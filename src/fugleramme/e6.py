"""The panel page packed for an external e-ink panel, so a microcontroller
can stream it to the glass without touching it.

The file carries only what decoding and checking the pixels takes. What a client
does between fetches - how long to sleep, the battery it reported - rides in the
response headers instead, so a stored file never holds stale advice.

A little-endian header, then the payload:

    0  magic "E6F1" (u32)       4  version (u16)        6  header length (u16)
    8  width (u16)              10 height (u16)         12 encoding (u8)
    13 reserved, zero (3 bytes) 16 payload length (u32) 20 payload CRC-32 (u32)

The payload starts at the header length, not at a fixed offset, and a client
ignores whatever it does not know: a field a client can skip is appended to the
header, which grows, while the version stays. The version moves only for a change
an old client would draw wrong.

Encoding 1 is the controllers' own colour codes, two pixels a byte with the left
one in the high nibble, in the order the glass's controllers read them. The
13.3" glass is driven as two halves: every row's left half goes to the first
controller (CS0), then every row's right half to the second (CS1), so the
payload is those two runs back to back. The 7.3" glass has one controller,
which reads its landscape rows top to bottom.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
from PIL import Image

from .config import PANEL_SIZES
from .panel import to_glass

MAGIC = b"E6F1"
VERSION = 2
ENCODING = 1
HEADER = struct.Struct("<4sHHHHB3xII")

# The glasses `encode` packs for: the 13.3" Seeed panel's, as two controllers,
# and the 7.3"'s, as one.
_TWO_CONTROLLERS = PANEL_SIZES["13.3"]
GLASSES = (PANEL_SIZES["7.3"], _TWO_CONTROLLERS)

# `dither`'s palette order - black, white, yellow, red, blue, green - to the
# controllers' codes, which skip 4.
_NATIVE = np.array([0, 1, 2, 3, 5, 6], dtype=np.uint8)


def encode(frame: Image.Image, rotation: int) -> bytes:
    """The dithered panel page, as it hangs, packed for the glass."""
    glass = to_glass(frame, rotation)
    halves = glass.size == _TWO_CONTROLLERS
    # The 13.3"'s controllers read it a quarter turn clockwise from the landscape
    # glass, as Pimoroni's driver turns it.
    scan = glass.transpose(Image.Transpose.ROTATE_270) if halves else glass
    pixels = np.asarray(scan, dtype=np.uint8)
    height, width = pixels.shape
    codes = _NATIVE[pixels]
    ordered = np.concatenate(np.hsplit(codes, 2)) if halves else codes
    payload = ((ordered[:, 0::2] << 4) | ordered[:, 1::2]).tobytes()
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER.size,
        width,
        height,
        ENCODING,
        len(payload),
        zlib.crc32(payload),
    )
    return header + payload
