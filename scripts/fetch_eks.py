from pathlib import Path
import re, shutil, subprocess

TMP = Path("/tmp/eks_clone")
DEST = Path("data/raw_docs/aws-eks")
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(DEST, ignore_errors=True)
DEST.mkdir(parents=True)

url = "https://" + "github.com/awsdocs/amazon-eks-user-guide.git"
subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(TMP)], check=True)
subprocess.run(["git", "-C", str(TMP), "sparse-checkout", "set", "latest/ug"], check=True)

for adoc in sorted((TMP / "latest/ug").rglob("*.adoc"))[:160]:
    text = adoc.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"^====\s+", "### ", text, flags=re.M)
    text = re.sub(r"^===\s+", "## ", text, flags=re.M)
    text = re.sub(r"^==\s+", "# ", text, flags=re.M)
    text = re.sub(r"^=\s+", "# ", text, flags=re.M)
    text = re.sub(r"^----\s*\$", "```", text, flags=re.M)
    out = DEST / adoc.relative_to(TMP / "latest/ug").with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

shutil.rmtree(TMP, ignore_errors=True)
print(f"Saved {len(list(DEST.rglob('*.md')))} AWS EKS docs!")
