"""ADF2016 (旧版) 识别、脚本格式自动转换、环境测试、无 PLAMS 读取结果；字体缩放。"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fake_adf2016(base):
    home = Path(base) / "ADF2016.101"
    (home / "bin").mkdir(parents=True)
    (home / "atomicdata").mkdir()
    (home / "license.txt").write_text("lic")
    (home / "bin" / "adf").write_text('''#!/bin/sh
inp=$(cat)
[ -z "$ADFRESOURCES" ] && exit 1
echo "$inp" | grep -q "End Input" || exit 3
if echo "$inp" | grep -q "^Fragments"; then
  for f in $(echo "$inp" | grep -o "[A-Za-z0-9_]*\\.t21"); do [ -f "$f" ] || exit 2; done
fi
echo x > TAPE21
''')
    (home / "bin" / "dmpkf").write_text('''#!/bin/sh
printf "TransferIntegrals\\nV(hole)\\n 1 1 2\\n  -2.679E-02\\nV(electron)\\n 1 1 2\\n   1.3129E-02\\n"
''')
    for f in ("adf", "dmpkf"):
        (home / "bin" / f).chmod(0o755)
    return home


def test_adf2016():
    if not (shutil.which("bash") or shutil.which("sh")):
        print("跳过 (无 sh)")
        return
    from mobility_tool.adf import generate_adf_inputs, read_adf_transfer_integrals, script_style
    from mobility_tool.programs import find_ams, run_adf_script, test_adf_environment
    with tempfile.TemporaryDirectory() as d:
        home = fake_adf2016(d)
        info = find_ams(d)                                    # 选上一级目录也能找到
        assert info and info["kind"] == "classic" and info["version"] == "2016.101", info
        assert test_adf_environment(info, Path(d) / "t")[0]
        r = generate_adf_inputs(Path(d) / "job", dimer=ROOT / "examples/benzene_dimer.xyz", n_a=12, style="ams")
        ok, log, msg = run_adf_script(info, r["script"], Path(d) / "scr")
        assert ok and "自动重新生成" in msg and script_style(r["script"]) == "classic", msg
        res = read_adf_transfer_integrals(Path(d) / "job" / "dimer_full.t21")
        assert abs(res["hole"]["V_meV"] + 728.99) < 0.1
        (Path(d) / "job" / "dimer_full.txt").unlink()
        res = read_adf_transfer_integrals(Path(d) / "job" / "dimer_full.t21", ams_info=info)
        assert abs(res["electron"]["V_meV"] - 357.26) < 0.1
        assert home.exists()
    print("ADF2016 OK: 识别版本 2016.101 (旧版)、环境测试、AMS 脚本自动转为旧版格式并运行、无 PLAMS 读取 TAPE21")


def test_font_scale():
    try:
        from mobility_tool import theme
        old = theme.SCALE
        theme.SCALE = 1.3
        assert theme.sz(10) == 13 and theme.sz(26) == 34 and theme.sz(4) == 7
        theme.SCALE = old
    except ModuleNotFoundError:          # 无 tkinter 的环境 (界面测试中已覆盖)
        pass
    from mobility_tool import i18n
    i18n.set_lang("en")
    assert i18n.tr("大") == "Large" and i18n.tr("最大原子偏移") != "最Large原子偏移"
    i18n.set_lang("zh")
    print("字体缩放 OK, 单字翻译只做整句匹配 OK")


if __name__ == "__main__":
    test_adf2016()
    test_font_scale()
