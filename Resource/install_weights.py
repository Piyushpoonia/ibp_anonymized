"""Verify and unpack the frozen CLIP and BLIP Hugging Face model cache."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARTS = tuple(ROOT / f"hf_models.part{index:02d}" for index in range(1, 4))
HUB = ROOT / "hf_home" / "hub"
EXPECTED_SHA256 = "5e9e1a2f942fbdb61065d21c0b9cd7dc600bcad204aa7035f5bf4db422aaae85"
MODELS = (
    "models--openai--clip-vit-base-patch16",
    "models--Salesforce--blip-image-captioning-base",
)


class PartReader(io.RawIOBase):
    """Expose the ordered archive chunks as one read-only stream."""

    def __init__(self, parts: tuple[Path, ...]):
        self.parts = parts
        self.index = 0
        self.current = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        while True:
            if self.current is None:
                if self.index == len(self.parts):
                    return 0
                self.current = self.parts[self.index].open("rb")
                self.index += 1
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def installed() -> bool:
    for name in MODELS:
        reference = HUB / name / "refs" / "main"
        if not reference.is_file():
            return False
        revision = reference.read_text(encoding="utf-8").strip()
        snapshot = HUB / name / "snapshots" / revision
        required = ("config.json", "preprocessor_config.json", "tokenizer.json", "pytorch_model.bin")
        if not all((snapshot / filename).is_file() for filename in required):
            return False
    return True


def main() -> None:
    if installed():
        print("Frozen CLIP and BLIP caches already installed.")
        return
    missing = [str(part) for part in PARTS if not part.is_file()]
    if missing:
        raise SystemExit("Missing frozen-weight archive parts:\n" + "\n".join(missing))
    digest = hashlib.sha256()
    for part in PARTS:
        with part.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs"):
                raise SystemExit(f"Git LFS pointer found instead of weight data: {part}; run git lfs pull")
            stream.seek(0)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise SystemExit("Frozen-weight archive SHA-256 mismatch; refusing to extract.")
    HUB.mkdir(parents=True, exist_ok=True)
    with PartReader(PARTS) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                if member.name.split("/", 1)[0] not in MODELS:
                    raise SystemExit(f"Unexpected path in frozen-weight archive: {member.name}")
                tar.extract(member, path=HUB, filter="data")
    if not installed():
        raise SystemExit("Archive extraction finished without both required model refs.")
    print(f"Frozen CLIP and BLIP caches installed under {HUB}")


if __name__ == "__main__":
    main()
