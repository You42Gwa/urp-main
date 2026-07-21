"""Download/load BLIP and caption one local raster image for environment verification."""

from pathlib import Path

from mm_gate.caption import BlipCaptioner


def main() -> None:
    image = next(Path("data/images").rglob("*.jpg"))
    captioner = BlipCaptioner(Path(".cache/huggingface"))
    print(image)
    print(captioner.device)
    print(captioner.caption(image))


if __name__ == "__main__":
    main()
