"""Hồ sơ mỗi lần chạy của mạch T2: lưu ĐỦ config + kết quả để truy ngược mọi con số.

Mỗi lần gọi scripts/t2/eval_robust.py tạo một thư mục run:
    <run_root>/<YYYYmmdd-HHMMSS>_<method>_<env>_seed<S>_N<n>_obs<o>_<mode>/
        run.yaml          lệnh chạy, tham số đã gộp, lưới σ, config checkpoint, tham số env
                          thực tế, git (commit, trạng thái), phiên bản thư viện, thiết bị,
                          trạng thái + thời gian, bảng tổng hợp kết quả
        ckpt_config.yaml  bản sao config.yaml của checkpoint
        inputs/           bản sao file --config / --grid đã dùng
        code/             bản sao code T2 lúc chạy + git_diff.patch (thay đổi chưa commit)
        episodes.csv      mỗi episode một dòng (analysis/robust_csv.py)
        summary.md / summary.csv   bảng tổng hợp theo σ (analysis/robust_summary.py)
        console.log       toàn bộ output console
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

# code T2 ảnh hưởng tới kết quả — chép nguyên văn vào mỗi run
CODE_FILES = [
    "envs/noise_wrapper.py",
    "analysis/robust_metrics.py",
    "analysis/robust_csv.py",
    "analysis/robust_summary.py",
    "analysis/run_record.py",
    "scripts/t2/eval_robust.py",
]
PACKAGES = ["jax", "jaxlib", "jax-cuda12-plugin", "flax", "optax", "jraph", "numpy",
            "tensorflow-probability", "dgppo"]


def _git(repo: Path, *args) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return out.stdout.strip()


def _plain(x: Any) -> Any:
    """Đổi về kiểu YAML thuần (dict/list/str/số/bool/None)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "item"):          # numpy / jax scalar
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    status = _git(repo, "status", "--porcelain")
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "describe": _git(repo, "describe", "--always", "--dirty"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "dgppo_commit": _git(repo / "third_party" / "dgppo", "rev-parse", "HEAD"),
    }


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class Tee:
    """Ghi stdout ra cả terminal lẫn file."""

    def __init__(self, path: Path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class RunRecord:
    def __init__(self, run_root: str, name: str, repo: Path):
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(run_root) / f"{stamp}_{name}"
        suffix = 1
        while self.dir.exists():   # hai run cùng giây
            self.dir = Path(run_root) / f"{stamp}-{suffix}_{name}"
            suffix += 1
        self.dir.mkdir(parents=True)
        self.repo = repo
        self.data: dict = {
            "run_dir": str(self.dir),
            "status": "running",
            "started": _dt.datetime.now().isoformat(timespec="seconds"),
            "command": [sys.executable] + sys.argv,
            "cwd": os.getcwd(),
            "host": socket.gethostname(),
            "env_vars": {k: os.environ.get(k) for k in
                         ("CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "JAX_PLATFORMS",
                          "XLA_PYTHON_CLIENT_PREALLOCATE", "PYTHONPATH")},
            "git": git_info(repo),
            "versions": versions(),
        }
        self._save_code()
        sys.stdout = Tee(self.dir / "console.log")

    def _save_code(self) -> None:
        code = self.dir / "code"
        code.mkdir()
        for rel in CODE_FILES:
            src = self.repo / rel
            if src.exists():
                dst = code / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        (code / "git_diff.patch").write_text(_git(self.repo, "diff", "HEAD") + "\n")

    def copy_input(self, path: str | None, label: str) -> None:
        if not path:
            return
        inputs = self.dir / "inputs"
        inputs.mkdir(exist_ok=True)
        shutil.copy2(path, inputs / f"{label}_{Path(path).name}")
        self.data.setdefault("inputs", {})[label] = {
            "path": str(Path(path).resolve()), "content": Path(path).read_text()}

    def set(self, key: str, value: Any) -> None:
        self.data[key] = _plain(value)
        self.save()

    def save(self) -> None:
        with open(self.dir / "run.yaml", "w") as f:
            yaml.safe_dump(_plain(self.data), f, sort_keys=False, allow_unicode=True, width=120)

    def finish(self, status: str = "done") -> None:
        self.data["status"] = status
        self.data["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.save()
        if isinstance(sys.stdout, Tee):
            sys.stdout.flush()
            sys.stdout = sys.stdout.stdout
