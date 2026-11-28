import hashlib
import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.core.schemas import DocMetadata, DocumentChunk

SERVICE_VERSIONS: dict[str, str] = {
    "kubernetes": "1.32",
    "aws-eks": "latest",
    "etcd-cncf": "v3.5",
}
MAX_CHUNK_CHARS = 1500


def _extract_frontmatter_and_body(text: str) -> tuple[str | None, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if not m:
        return None, text
    fm, body = m.group(1), text[m.end() :]
    t_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', fm, flags=re.MULTILINE)
    return (t_match.group(1).strip() if t_match else None), body


def _split_oversized_fence(fence_block: str, max_chars: int) -> list[str]:
    lines = fence_block.splitlines()
    if len(lines) < 3 or len(fence_block) <= max_chars:
        return [fence_block]
    open_fence, close_fence = lines[0], lines[-1]
    inner_lines = lines[1:-1]
    parts: list[str] = []
    buf: list[str] = []
    cur_len = len(open_fence) + len(close_fence) + 4
    for ln in inner_lines:
        if buf and cur_len + len(ln) + 1 > max_chars:
            parts.append("\n".join([open_fence, *buf, close_fence]))
            buf, cur_len = [ln], len(open_fence) + len(close_fence) + len(ln) + 5
        else:
            buf.append(ln)
            cur_len += len(ln) + 1
    if buf:
        parts.append("\n".join([open_fence, *buf, close_fence]))
    return parts


def _split_preserving_code_blocks(
    text: str, max_chars: int = MAX_CHUNK_CHARS
) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    tokens = re.split(r"(```[\s\S]*?```)", text)
    chunks: list[str] = []
    current = ""
    for token in tokens:
        if not token.strip():
            continue
        is_code = token.startswith("```") and token.endswith("```")
        units = (
            _split_oversized_fence(token.strip(), max_chars)
            if is_code
            else [p.strip() for p in token.split("\n\n") if p.strip()]
        )
        for unit in units:
            if len(unit) > max_chars and not is_code:
                for i in range(0, len(unit), max_chars):
                    if current.strip():
                        chunks.append(current.strip())
                        current = ""
                    chunks.append(unit[i : i + max_chars].strip())
                continue
            if current and len(current) + len(unit) + 2 > max_chars:
                chunks.append(current.strip())
                current = unit
            else:
                current = f"{current}\n\n{unit}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_markdown_file(file_path: Path, raw_root: Path) -> list[DocumentChunk]:
    rel_path = file_path.relative_to(raw_root)
    service_key = rel_path.parts[0]
    if service_key not in SERVICE_VERSIONS:
        return []
    title, body = _extract_frontmatter_and_body(
        file_path.read_text(encoding="utf-8", errors="ignore")
    )
    if len(body.strip()) < 60:
        return []
    if title and not re.match(r"^\s*#\s+", body):
        body = f"# {title}\n\n{body}"

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    chunks: list[DocumentChunk] = []
    ordinal = 0
    for doc in splitter.split_text(body):
        parts = [doc.metadata[k] for k in ("h1", "h2", "h3") if doc.metadata.get(k)]
        section = " > ".join(parts) if parts else (title or file_path.stem)
        for sub_text in _split_preserving_code_blocks(doc.page_content):
            if len(sub_text) < 40:
                continue
            ctx_text = f"[{service_key.upper()} | {rel_path.as_posix()} | {section}]\n{sub_text}"
            content_sha = hashlib.sha1(sub_text.encode()).hexdigest()[:12]
            digest = hashlib.sha256(
                f"{rel_path.as_posix()}:{ordinal}:{content_sha}".encode()
            ).hexdigest()[:32]
            uid = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
            ordinal += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=uid,
                    content=ctx_text,
                    metadata=DocMetadata(
                        service=service_key,  # type: ignore[arg-type]
                        version=SERVICE_VERSIONS[service_key],
                        file_path=rel_path.as_posix(),
                        section_hierarchy=section,
                        has_code_block="```" in sub_text,
                    ),
                )
            )
    return chunks


def load_and_chunk_corpus(
    raw_root: Path = Path("data/raw_docs"),
) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for service in SERVICE_VERSIONS:
        service_dir = raw_root / service
        if not service_dir.exists():
            continue
        md_files = sorted(service_dir.rglob("*.md"))
        service_chunks = [c for f in md_files for c in chunk_markdown_file(f, raw_root)]
        print(
            f"Chunked {len(md_files)} files -> {len(service_chunks)} chunks for [{service}]"
        )
        all_chunks.extend(service_chunks)
    return all_chunks
