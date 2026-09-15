"""文件选择窗口。

当前只负责选择 DXF，不负责读取、识别或修改文件。
"""

from __future__ import annotations

from pathlib import Path


def choose_dxf_file(initial_dir: Path) -> Path | None:
    """打开 DXF 文件选择窗口；取消时返回 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("当前 Python 环境没有可用的 tkinter，无法打开文件选择窗口。") from error

    initial_dir = initial_dir.expanduser().resolve()
    if not initial_dir.is_dir():
        initial_dir = Path(__file__).resolve().parent

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title="选择待处理的 DXF 文件",
            initialdir=str(initial_dir),
            filetypes=[("DXF 文件", "*.dxf")],
        )
    finally:
        root.destroy()

    return Path(selected).expanduser().resolve() if selected else None
