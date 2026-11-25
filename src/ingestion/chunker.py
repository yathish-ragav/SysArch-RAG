import hashlib
from pathlib import Path
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.core.schemas import DocMetadata, DocumentChunk

SERVICE_VERSIONS: dict[str, str] = {
    "kubernetes": "1.31",
    "aws-eks": "latest",
    "etcd-cncf": "v3.5",
}
MAX_CHUNK_CHARS = 2400
OVERLAP_CHARS = 250


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)


def _split_preserving_code_blocks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    tokens = re.split(r"(```[\s\(\S\)]*?```)", text)
    chunks: list[str] = []
    current = ""
    for token in tokens:
        if not token.strip():
            continue
        is_code = token.startswith("```") and token.endswith("```")
        if is_code:
            if len(current) + len(token) > max_chars and current.strip():
                chunks.append(current.strip())
                current = token
            else:
                current = f"{current}\n\n{token}".strip()
        else:
            for para in token.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 > max_chars and current.strip():
                    chunks.append(current.strip())
                    tail = current[-OVERLAP_CHARS:] if not current.endswith("```") else ""
                    current = f"{tail}\n\n{para}".strip()
                else:
                    current = f"{current}\n\n{para}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_markdown_file(file_path: Path, raw_root: Path) -> list[DocumentChunk]:
    rel_path = file_path.relative_to(raw_root)
    service_key = rel_path.parts[0]
    if service_key not in SERVICE_VERSIONS:
        return []
    clean_text = _strip_frontmatter(file_path.read_text(encoding="utf-8", errors="ignore"))
    if len(clean_text.strip()) < 80:
        return []
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    chunks: list[DocumentChunk] = []
    for doc in splitter.split_text(clean_text):
        parts = [doc.metadata[k] for k in ("h1", "h2", "h3") if doc.metadata.get(k)]
        section = " > ".join(parts) if parts else file_path.stem
        for idx, sub_text in enumerate(_split_preserving_code_blocks(doc.page_content)):
            if len(sub_text) < 60:
                continue
            ctx_text = f"[{service_key.upper()} | {rel_path.as_posix()} | {section}]\n{sub_text}"
            digest = hashlib.sha256(f"{rel_path.as_posix()}:{section}:{idx}".encode()).hexdigest()[:32]
            uid = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
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


def load_and_chunk_corpus(raw_root: Path = Path("data/raw_docs"), max_files_per_service: int = 150) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for service in SERVICE_VERSIONS:
        service_dir = raw_root / service
        if not service_dir.exists():
            continue
        md_files = sorted(service_dir.rglob("*.md"))[:max_files_per_service]
        service_chunks = [c for f in md_files for c in chunk_markdown_file(f, raw_root)]
        print(f"Chunked {len(md_files)} files -> {len(service_chunks)} chunks for [{service}]")
        all_chunks.extend(service_chunks)
    return all_chunks
