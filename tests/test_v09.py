"""中文 Windows (GBK) 下的文本读写、ADF 批量计算 (片段共用 / 断点续算 / 自动读取)。"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import mobility_tool  # noqa: E402,F401  (安装文本编码处理)


def test_text_encoding():
    with tempfile.TemporaryDirectory() as d:
        u8, gbk, crlf = Path(d) / "u8.sh", Path(d) / "old.run", Path(d) / "win.txt"
        u8.write_bytes("#!/bin/sh\n# ADF 转移积分 (ADF2019)\n# 结果文件: x.t21\n".encode("utf-8"))   # 截图中报错的文件
        gbk.write_bytes("#! /bin/sh\n# 需先在 ADFjobs 中运行 mono.run\n".encode("gbk"))
        crlf.write_bytes("第一行\r\n第二行\r\n".encode("gbk"))
        assert "转移积分" in u8.read_text() and "需先在" in gbk.read_text()
        assert crlf.read_text() == "第一行\n第二行\n"
        from mobility_tool.programs import adf_result_files
        assert adf_result_files(u8)[0].name == "x.t21"           # 以前在 GBK 下会识别不出结果文件标记
        p = Path(d) / "w.json"
        p.write_text('{"路径": "C:/数据"}')
        assert p.read_bytes().decode("utf-8")
    print("文本编码 OK: UTF-8 / GBK / CRLF 均正确读取，写入统一为 UTF-8")


def test_batch():
    if not (shutil.which("bash") or shutil.which("sh")):
        print("跳过 (无 sh)")
        return
    from test_crystal import make_cif
    from test_v08 import fake_adf2016
    from mobility_tool.adf import _same_fragment
    from mobility_tool.crystal import find_dimers, find_molecules, read_cif
    from mobility_tool.programs import adf_result_files, find_ams, run_adf_script
    from mobility_tool.workflow import collect_group_couplings, prepare_dimer_jobs
    with tempfile.TemporaryDirectory() as d:
        fake_adf2016(d)
        make_cif(f"{d}/t.cif")
        c = read_cif(f"{d}/t.cif")
        m = find_molecules(c)
        r = find_dimers(c, m, cutoff=4.5)
        a = [(s, *x) for s, x in r["atoms_a"]]
        assert _same_fragment(a, [(s, x + 3, y - 1, z) for s, x, y, z in a])       # 平移
        mirror = [(s, -x, y, z) for s, x, y, z in a]
        import numpy as np
        if np.ptp(np.array([x[1:] for x in a]) @ np.linalg.svd(np.array([x[1:] for x in a]) -
                                                                np.array([x[1:] for x in a]).mean(0))[2][2]) > 0.1:
            assert not _same_fragment(a, mirror)                                      # 非平面分子的镜像不能共用
        out = Path(d) / "job"
        prepare_dimer_jobs(r, out, "adf", style="classic", chunk=3)
        scripts = sorted(out.glob("batch_adf_*.sh"))
        assert len(scripts) == 2 and all(s_.with_suffix(".run").exists() for s_ in scripts)
        info = find_ams(d)
        for s_ in scripts:
            ok, log, msg = run_adf_script(info, s_)
            assert ok, msg
        coup, err = collect_group_couplings(out, "adf", ams_info=info)
        assert len(coup) == len(r["groups"]) and not err
        res = [x for s_ in scripts for x in adf_result_files(s_)]
        res[0].unlink()
        ok, log, msg = run_adf_script(info, scripts[0])
        assert ok and log.read_text().count("running") == 1                           # 断点续算: 只算缺的一个
    print(f"ADF 批量 OK: {len(r['groups'])} 组分 2 个批量脚本 (chunk=3)，片段共用、续算、自动读取均正确")


if __name__ == "__main__":
    test_text_encoding()
    test_batch()
