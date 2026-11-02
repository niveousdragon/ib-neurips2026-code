"""Scan the repository for things that must not be published.

Looks for absolute machine paths, user names in paths, credentials and tokens,
and oversized files. Exit code 1 if anything is found, so it can gate a commit.

    python tools/check_release.py [ROOT]
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ruff_cache", ".pytest_cache", "data"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".csv", ".json",
                 ".cff", ".ini", ".sh", ".bat", ".ps1", ""}

PATTERNS = {
    "windows absolute path": re.compile(r"\b[A-Za-z]:[\\/]{1,2}(Users|ProgramData|Program Files)", re.I),
    "home directory path": re.compile(r"(/home/|/Users/)[A-Za-z0-9._-]+/"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "aws key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "slack/anthropic/openai key": re.compile(r"\b(xox[abp]-|sk-ant-|sk-[A-Za-z0-9]{32,})"),
    "credential assignment": re.compile(
        r"(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]", re.I),
    "e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
# Names of the machine and internal project folders the code was developed in.
WORDS = re.compile(r"PycharmProjects|flt-workspace|mango_data|rebuttal|camera[_ -]?ready", re.I)


def main() -> int:
    problems = []
    for path in sorted(ROOT.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        rel = path.relative_to(ROOT)
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: file larger than {MAX_BYTES // 2**20} MB")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS.items():
                if pat.search(line):
                    problems.append(f"{rel}:{lineno}: {name}: {line.strip()[:120]}")
            if WORDS.search(line):
                problems.append(f"{rel}:{lineno}: internal name: {line.strip()[:120]}")
    for p in problems:
        print(p)
    print(f"[{'FAILED' if problems else 'OK'}] {len(problems)} finding(s) in {ROOT}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
