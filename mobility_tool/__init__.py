"""Mobilis（迁跃）：有机半导体电荷输运与光物理计算平台"""
import pathlib as _pathlib

# ---------------------------------------------------------------------------------------------
# 文本编码：中文 Windows 的默认编码是 GBK，而本程序写出的脚本 / 配置为 UTF-8，
# 未指定编码时读写会出现 "'gbk' codec can't decode ..."。这里统一为：
#   写：未指定编码时用 UTF-8；读：依次尝试 UTF-8 (含 BOM) → GBK → UTF-8 (替换无法解码的字符)。
# 只影响未显式指定 encoding 的调用，与 Python 的 UTF-8 模式 (PEP 686) 行为一致。
# ---------------------------------------------------------------------------------------------
_orig_read_text = _pathlib.Path.read_text
_orig_write_text = _pathlib.Path.write_text


def _read_text(self, encoding=None, errors=None, *args, **kw):
    if encoding is not None:
        return _orig_read_text(self, encoding, errors, *args, **kw)
    data = self.read_bytes()
    for enc in ("utf-8-sig", "gbk"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors or "replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _write_text(self, data, encoding=None, errors=None, *args, **kw):
    return _orig_write_text(self, data, encoding or "utf-8", errors, *args, **kw)


if not getattr(_pathlib.Path.read_text, "_mobilis", False):
    _read_text._mobilis = True
    _pathlib.Path.read_text = _read_text
    _pathlib.Path.write_text = _write_text


def _safe_console():
    """中文 Windows 控制台为 GBK，不能输出 Å、²、⟨、∇ 等字符；改为用 ? 代替而不是抛出 UnicodeEncodeError。"""
    import sys
    for st in (sys.stdout, sys.stderr):
        enc = (getattr(st, "encoding", None) or "").lower().replace("-", "")
        if enc and enc != "utf8" and hasattr(st, "reconfigure"):
            try:
                st.reconfigure(errors="replace")
            except Exception:                      # noqa: BLE001  (被替换的流可能不支持)
                pass


_safe_console()

from .mobility import compute_mobility, marcus_rate
from .reorganization import analyze_reorg, generate_reorg_inputs
from .transfer_integral import calc_transfer_integrals, generate_ti_inputs

__version__ = "1.20.2"
__author__ = "Yarui Shi"
__affiliation__ = "安阳师范学院 (Anyang Normal University)"
