"""No mixed Chinese/English output (v1.29.1). Every message and help text of the program is checked in both interface
languages: with English, no Chinese may remain after translation; with Chinese, no English sentence may remain.
Source strings are collected from the code (docstrings, file contents written for other programs and data columns
excluded), so a new message without a translation makes this test fail."""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HAN = re.compile(r"[一-鿿]")
SENT = re.compile(r"\b[a-z]{3,}\b(?:[ ,;:]+\b[a-z]{2,}\b){2,}")       # three or more lower-case English words
CALLS = {"print", "progress", "on_progress", "report", "SystemExit", "ValueError", "RuntimeError", "FileNotFoundError",
         "KeyError", "FileExistsError", "add_argument", "add_parser", "warn", "warning", "ArgumentParser"}
SKIP_FILES = {"_cli_i18n.py", "_cli_i18n_more.py", "_i18n_table.py", "i18n.py", "_sto6g.py"}
FILE_CONTENT = re.compile(r"^\s*(#|rem |echo |where |@echo|if not|set |cd |%|\")|\\s|_黄昆因子|_矩阵|^中文$|^语言 / Language$")
TECH = re.compile(r"(--[\w-]+|\S+\.(?:py|csv|json|xyz|cif|gjf|inp|out|log|fchk|t21|rkf|bat|sh|npz)\b|\b[A-Z][\w=/-]*=\S+|"
                  r"python -m \S+(?: \S+)*|pip install \S+|gmx \S+(?: -\S+ \S+)*|<[^>]+>|'[^']*'|\"[^\"]*\")")


def fname(c):
    f = c.func
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")


def strings():
    zh, en = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "mobility_tool", "*.py"))):
        if os.path.basename(f) in SKIP_FILES:
            continue
        tree = ast.parse(open(f, encoding="utf-8").read())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and HAN.search(n.value) \
                    and not FILE_CONTENT.search(n.value):
                zh.append((os.path.basename(f), n.value))
            if isinstance(n, ast.Call) and fname(n) in CALLS:
                args = list(n.args) + [k.value for k in n.keywords if k.arg in ("help", "description")]
                for a in args:
                    for m in ast.walk(a):
                        if isinstance(m, ast.Constant) and isinstance(m.value, str) and not HAN.search(m.value):
                            en.append((os.path.basename(f), m.value))
    return zh, en


def main():
    from mobility_tool import i18n
    zh, en = strings()
    i18n.CLI_LANG = "en"
    bad_zh = sorted({s for _, s in zh if HAN.search(i18n.tr_cli(s))})
    i18n.CLI_LANG = "zh"
    bad_en = sorted({s for _, s in en if SENT.search(TECH.sub(" ", i18n.tr_cli(s)))})
    from mobility_tool._cli_i18n import OPTION_HELP
    bad_en += sorted({v for v in OPTION_HELP.values() if i18n.tr_cli(v) == v})
    i18n.CLI_LANG = "en"
    for s in bad_zh[:40]:
        print("zh untranslated:", repr(s))
    for s in bad_en[:40]:
        print("en untranslated:", repr(s))
    assert not bad_zh and not bad_en, (len(bad_zh), len(bad_en))
    # end to end: the transport-regime diagnosis and two help pages in both languages
    import subprocess
    cif = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")
    runs = [["diagnose", cif, "--lam-h", "0.14", "--lam-e", "0.2", "--no-phonons", "--uq", "--uq-samples", "200"],
            ["scan", "level", "--help"], ["ceds", "--help"]]
    for lang in ("en", "zh"):
        for args in runs:
            env = dict(os.environ, MOBILIS_LANG=lang, PYTHONIOENCODING="utf-8")
            out = subprocess.run([sys.executable, "-m", "mobility_tool"] + args, cwd=ROOT, capture_output=True,
                                 text=True, encoding="utf-8", env=env, timeout=600).stdout
            assert out.strip(), args
            for ln in out.splitlines():
                if "{" in ln and "," in ln and "}" in ln:            # argparse lists of choices
                    continue
                if lang == "en":
                    assert not HAN.search(ln), (args[0], ln)
                else:
                    assert not SENT.search(TECH.sub(" ", ln)), (args[0], ln)
    print(f"{len(zh)} Chinese and {len(en)} English source strings, all translated")


if __name__ == "__main__":
    main()
    print("test_v50 ok")
