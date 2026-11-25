import json
from pathlib import Path
import re
import shutil
import subprocess

RAW_DIR = Path("data/raw_docs")
TMP_DIR = Path("/tmp/sysarch_doc_clones")
EXT_LANG = {".yaml": "yaml", ".yml": "yaml", ".json": "json", ".sh": "bash", ".go": "go", ".py": "python"}


def _clone(owner: str, repo: str, dest: Path, paths: list[str], branch: str | None = None) -> str:
    url = "/".join(["https:/", "github.com", owner, f"{repo}.git"])
    cmd = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse"]
    if branch:
        cmd.extend(["--branch", branch])
    subprocess.run([*cmd, url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *paths], check=True)
    return subprocess.check_output(["git", "-C", str(dest), "rev-parse", "--short", "HEAD"], text=True).strip()


def _resolve_k8s_shortcodes(text: str, examples_root: Path) -> str:
    def _embed_sample(match: re.Match[str]) -> str:
        rel = match.group(1).lstrip("/")
        sample_file = examples_root / rel
        if not sample_file.exists():
            return ""
        lang = EXT_LANG.get(sample_file.suffix.lower(), "yaml")
        code = sample_file.read_text(encoding="utf-8", errors="ignore").strip()
        return f"\n```{lang}\n{code}\n```\n"

    text = re.sub(r'\{\{[<%]\s*(?:code_sample|codenew)\s+file="([^"]+)"[^%<>]*[%>]\}\}', _embed_sample, text)
    text = re.sub(r'\{\{%\s*heading\s+"([^"]+)"\s*%\}\}', r"\1", text)
    text = re.sub(r'\{\{<\s*(?:note|caution|warning)\s*>\}\}', "\n> **Note:** ", text)
    text = re.sub(r'\{\{</\s*(?:note|caution|warning)\s*>\}\}', "\n", text)
    return text


def _adoc_to_md(text: str) -> str:
    out: list[str] = []
    in_code, code_lang = False, ""
    for line in text.splitlines():
        m_src = re.match(r"^\[source(?:,\s*([a-zA-Z0-9_-]+))?\]", line.strip())
        if m_src:
            code_lang = m_src.group(1) or "bash"
            continue
        if line.strip() == "----":
            in_code = not in_code
            out.append(f"```{code_lang}" if in_code else "```")
            if not in_code:
                code_lang = ""
            continue
        if line.strip() == "====":
            continue
        if not in_code:
            line = re.sub(r"^====\s+(.+)$", r"### \1", line)
            line = re.sub(r"^===\s+(.+)$", r"## \1", line)
            line = re.sub(r"^==\s+(.+)$", r"# \1", line)
            line = re.sub(r"^=\s+(.+)$", r"# \1", line)
        out.append(line)
    return "\n".join(out)


def main() -> None:
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True)
    manifest: dict[str, dict] = {}

    # 1. Kubernetes 1.32 docs + examples
    k8s_tmp = TMP_DIR / "k8s"
    k8s_sha = _clone(
        "kubernetes", "website",
        k8s_tmp,
        ["content/en/docs/concepts", "content/en/docs/tasks", "content/en/examples"],
        branch="release-1.32",
    )
    k8s_dest = RAW_DIR / "kubernetes"
    shutil.rmtree(k8s_dest, ignore_errors=True)
    for sub in ("concepts", "tasks"):
        src_root = k8s_tmp / "content/en/docs" / sub
        for md in src_root.rglob("*.md"):
            target = k8s_dest / sub / md.relative_to(src_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved = _resolve_k8s_shortcodes(md.read_text(encoding="utf-8", errors="ignore"), k8s_tmp / "content/en/examples")
            target.write_text(resolved, encoding="utf-8")
    manifest["kubernetes"] = {"sha": k8s_sha, "files": len(list(k8s_dest.rglob("*.md")))}

    # 2. AWS EKS User Guide (all latest/ug/*.adoc)
    eks_tmp = TMP_DIR / "eks"
    eks_sha = _clone("awsdocs", "amazon-eks-user-guide", eks_tmp, ["latest/ug"])
    eks_dest = RAW_DIR / "aws-eks"
    shutil.rmtree(eks_dest, ignore_errors=True)
    for adoc in (eks_tmp / "latest/ug").rglob("*.adoc"):
        target = eks_dest / adoc.relative_to(eks_tmp / "latest/ug").with_suffix(".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_adoc_to_md(adoc.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
    manifest["aws-eks"] = {"sha": eks_sha, "files": len(list(eks_dest.rglob("*.md")))}

    # 3. etcd v3.5 Operator Docs (etcd-io/website)
    etcd_tmp = TMP_DIR / "etcd"
    etcd_sha = _clone("etcd-io", "website", etcd_tmp, ["content/en/docs/v3.5"])
    etcd_dest = RAW_DIR / "etcd-cncf"
    shutil.rmtree(etcd_dest, ignore_errors=True)
    for md in (etcd_tmp / "content/en/docs/v3.5").rglob("*.md"):
        target = etcd_dest / md.relative_to(etcd_tmp / "content/en/docs/v3.5")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(md.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    manifest["etcd-cncf"] = {"sha": etcd_sha, "files": len(list(etcd_dest.rglob("*.md")))}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("MANIFEST:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
