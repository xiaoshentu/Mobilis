"""界面中英文切换。tr() 对整段文字按“最长片段优先”替换，因此也适用于由 f-string 拼出的文字。"""
import re

LANG = "zh"
_HAN = re.compile(r"[\u4e00-\u9fff]")
_SORTED = None


def set_lang(lang):
    global LANG
    LANG = "en" if str(lang).lower().startswith("en") else "zh"


def tr(text):
    if LANG == "zh" or not isinstance(text, str) or not _HAN.search(text):
        return text
    global _SORTED
    if _SORTED is None:
        from ._cli_i18n_more import ZH_EN_MORE
        from ._i18n_table import ZH_EN
        _SORTED = sorted({**ZH_EN_MORE, **ZH_EN}.items(), key=lambda kv: -len(kv[0]))
    if text in dict(_SORTED):
        return dict(_SORTED)[text]
    for zh, en in _SORTED:
        if len(zh) < 2:                  # 单字只做整句匹配，避免误改其它文字
            continue
        if zh in text:
            text = text.replace(zh, en)
            if not _HAN.search(text):
                break
    return text


# ---- command line: English by default, Chinese with MOBILIS_LANG=zh or the GUI setting ----
CLI_LANG = "en"
_CLI_SORTED = None


def cli_lang_from_env():
    """MOBILIS_LANG (zh / en) first, then the language saved by the GUI (~/.mobility_tool.json), else English."""
    import json
    import os
    from pathlib import Path
    v = os.environ.get("MOBILIS_LANG", "")
    if not v:
        try:
            v = json.loads((Path.home() / ".mobility_tool.json").read_text(encoding="utf-8")).get("lang", "")
        except Exception:
            v = ""
    return "zh" if str(v).lower().startswith("zh") else "en"


_EN_SORTED = None
_LATIN = re.compile(r"[A-Za-z]{2}")


def tr_cli_zh(text):
    """Chinese interface: English fragments of messages and help texts written in English (EN_ZH) → Chinese."""
    if not isinstance(text, str) or not _LATIN.search(text):
        return text
    global _EN_SORTED
    if _EN_SORTED is None:
        from ._cli_i18n_more import EN_ZH
        _EN_SORTED = (dict(EN_ZH), sorted(EN_ZH.items(), key=lambda kv: -len(kv[0])))
    table, items = _EN_SORTED
    if text in table:
        return table[text]
    for en, zh in items:
        if en in text:
            text = text.replace(en, zh)
    return text


def tr_cli(text):
    """Translate command-line help and messages (longest fragment first, CLI table before the GUI table).
    English interface: Chinese source fragments → English; Chinese interface: English fragments → Chinese."""
    if CLI_LANG == "zh":
        return tr_cli_zh(text)
    if not isinstance(text, str) or not _HAN.search(text):
        return text
    global _CLI_SORTED
    if _CLI_SORTED is None:
        from ._cli_i18n import CLI_ZH_EN
        from ._cli_i18n_more import ZH_EN_MORE
        from ._i18n_table import ZH_EN
        table = dict(ZH_EN)
        table.update(ZH_EN_MORE)
        table.update(CLI_ZH_EN)
        _CLI_SORTED = (table, sorted(table.items(), key=lambda kv: -len(kv[0])))
    table, items = _CLI_SORTED
    if text in table:
        return table[text]
    for zh, en in items:
        if len(zh) < 2:
            continue
        if zh in text:
            text = text.replace(zh, en)
            if not _HAN.search(text):
                break
    for a, b in (("；", "; "), ("，", ", "), ("：", ": "), ("（", " ("), ("）", ")"), ("。", ". "), ("、", ", ")):
        text = text.replace(a, b)
    return text


class TranslatingStream:
    """Wraps sys.stdout / sys.stderr so that every printed message appears in the interface language."""

    def __init__(self, stream):
        self._s = stream

    def write(self, s):
        return self._s.write(tr_cli(s))

    def __getattr__(self, name):
        return getattr(self._s, name)
