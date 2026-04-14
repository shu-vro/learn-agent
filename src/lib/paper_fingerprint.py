import hashlib
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.config.env import DEFAULT_DOWNLOADS_DIR


@dataclass(slots=True, frozen=True)
class PaperFingerprint:
    source: str
    local_path: Path
    sha256: str


def _is_remote_source(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _sha256_for_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:
        import certifi  # type: ignore

        context.load_verify_locations(cafile=certifi.where())
    except Exception:
        # Use system trust store when certifi is unavailable.
        pass
    return context


def _download_remote_source(source: str, download_dir: str | Path) -> Path:
    target_dir = Path(download_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(source)
    source_name = Path(parsed.path).name or "paper.pdf"
    if not Path(source_name).suffix:
        source_name = f"{source_name}.pdf"

    source_prefix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    local_path = target_dir / f"{source_prefix}-{source_name}"

    # Reuse the previously downloaded file for stable hashing and faster retries.
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    request = Request(source, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with (
            urlopen(request, timeout=120, context=_build_ssl_context()) as response,
            local_path.open("wb") as output_file,
        ):
            while chunk := response.read(1024 * 1024):
                output_file.write(chunk)
    except URLError as err:
        raise RuntimeError(
            "Failed to download remote paper source. "
            "If this is an SSL issue, install/update certifi or pass a local PDF path with --source. "
            f"source={source}"
        ) from err

    return local_path


def resolve_source_to_local_file(
    source: str,
    download_dir: str | Path = DEFAULT_DOWNLOADS_DIR,
) -> Path:
    if _is_remote_source(source):
        return _download_remote_source(source, download_dir)

    local_path = Path(source).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Paper source not found: {source}")
    return local_path


def fingerprint_paper_source(
    source: str,
    download_dir: str | Path = DEFAULT_DOWNLOADS_DIR,
) -> PaperFingerprint:
    local_path = resolve_source_to_local_file(source, download_dir)
    return PaperFingerprint(
        source=source,
        local_path=local_path,
        sha256=_sha256_for_file(local_path),
    )
