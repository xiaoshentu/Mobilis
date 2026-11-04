#!/bin/bash
# 依次运行重组能 / IP / EA 计算（可按需改为提交到作业系统）
g16 < anthracene_neutral.gjf > anthracene_neutral.log
formchk anthracene_neutral.chk anthracene_neutral.fchk
g16 < anthracene_cation.gjf > anthracene_cation.log
formchk anthracene_cation.chk anthracene_cation.fchk
g16 < anthracene_anion.gjf > anthracene_anion.log
formchk anthracene_anion.chk anthracene_anion.fchk
# *_neutral.fchk / *_cation.fchk / *_anion.fchk 含 Hessian，可用于重组能分解
