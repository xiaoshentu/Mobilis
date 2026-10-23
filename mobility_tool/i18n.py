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
        from ._i18n_table import ZH_EN
        _SORTED = sorted(ZH_EN.items(), key=lambda kv: -len(kv[0]))
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
