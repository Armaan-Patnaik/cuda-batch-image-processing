#!/usr/bin/env python3

import argparse
from pathlib import Path


def generate_image(index: int, width: int, height: int) -> bytes:
    pixels = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            gradient = (2 * x + 3 * y + 7 * index) % 256

            dx = x - width // 2
            dy = y - height // 2
            circle = (dx * dx + dy * dy) // max(1, width // 8)

            pixels[y * width + x] = (gradient + circle) % 256

    return bytes(pixels)


def write_pgm(path: Path, width: int, height: int, pixels: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate many small PGM images for the CUDA batch image project."
    )
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.count):
        pixels = generate_image(
            index,
            args.width,
            args.height,
        )
        write_pgm(
            output_dir / f"image_{index:04d}.pgm",
            args.width,
            args.height,
            pixels,
        )

    print(
        f"Generated {args.count} images "
        f"of size {args.width}x{args.height} in {output_dir}"
    )


if __name__ == "__main__":
    main()
