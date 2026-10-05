"""Shim tương thích JAX >=0.6 cho DGPPO (pin tại 51b3b11, viết cho jax 0.4.x).

Tự động nạp khi thư mục này nằm trong PYTHONPATH (Python import sitecustomize
lúc khởi động). Khôi phục các alias top-level jax.tree_* đã bị bỏ ở JAX 0.6.0
-> ánh xạ về jax.tree_util.*. Không đụng mã nguồn submodule.
"""
try:
    import jax
    import jax.tree_util as _jtu

    for _name in (
        "tree_map", "tree_multimap", "tree_leaves", "tree_flatten",
        "tree_unflatten", "tree_structure", "tree_transpose",
        "tree_reduce", "tree_all",
    ):
        try:
            _has = hasattr(jax, _name)
        except Exception:
            _has = False  # 0.6 raise AttributeError khi truy cập alias đã bỏ
        if not _has and hasattr(_jtu, _name):
            setattr(jax, _name, getattr(_jtu, _name))
except Exception:
    pass

# --- ffmpeg cho render video (hệ thống không có ffmpeg) ---
# Trỏ matplotlib vào binary ffmpeg tĩnh của imageio-ffmpeg để FFMpegWriter ghi .mp4
# được (nếu không, matplotlib fallback PillowWriter -> lỗi "unknown file extension").
try:
    import os
    import imageio_ffmpeg
    _exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _exe)
    import matplotlib
    matplotlib.rcParams["animation.ffmpeg_path"] = _exe
except Exception:
    pass
