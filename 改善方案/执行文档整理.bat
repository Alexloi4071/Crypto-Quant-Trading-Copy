@echo off
chcp 65001 >nul
echo ========================================
echo 改善方案文档整理脚本
echo ========================================

cd /d "%~dp0"

echo.
echo [1/5] 创建归档目录...
if not exist archive mkdir archive
echo ✅ 归档目录已创建

echo.
echo [2/5] 移动过时文档到归档...
if exist "00_综合评估报告.md" move "00_综合评估报告.md" archive\ >nul
if exist "01_核心改善方案.md" move "01_核心改善方案.md" "archive\详细方案_01_核心.md" >nul
if exist "02_模型與回測.md" move "02_模型與回測.md" "archive\详细方案_02_模型.md" >nul
if exist "03_工程化與實施.md" move "03_工程化與實施.md" "archive\详细方案_03_工程.md" >nul
if exist "04_补充改善方案_技术指标与UI.md" move "04_补充改善方案_技术指标与UI.md" "archive\详细方案_04_补充.md" >nul
echo ✅ 详细方案已归档

echo.
echo [3/5] 移动研究报告和路线图...
if exist "05_技术指标参数研究报告.md" move "05_技术指标参数研究报告.md" "archive\技术指标研究_完整版.md" >nul
if exist "实施路线图.md" move "实施路线图.md" "archive\详细实施路线图.md" >nul
if exist "总结与回应.md" move "总结与回应.md" "archive\总结与回应_原版.md" >nul
echo ✅ 研究报告已归档

echo.
echo [4/5] 删除重复文档...
if exist "START_HERE.md" del "START_HERE.md"
if exist "文档整理方案.md" move "文档整理方案.md" archive\ >nul
echo ✅ 重复文档已删除

echo.
echo [5/5] 重命名核心文档...
if exist "精简版改善方案_修改为主.md" ren "精简版改善方案_修改为主.md" "实施方案.md.tmp" >nul
if exist "实施方案.md.tmp" ren "实施方案.md.tmp" "实施方案.md" >nul

if exist "精确修改指南.md" ren "精确修改指南.md" "修改操作手册.md.tmp" >nul
if exist "修改操作手册.md.tmp" ren "修改操作手册.md.tmp" "修改操作手册.md" >nul

if exist "快速参考卡.md" ren "快速参考卡.md" "快速参考.md.tmp" >nul
if exist "快速参考.md.tmp" ren "快速参考.md.tmp" "快速参考.md" >nul

echo ✅ 核心文档已重命名

echo.
echo ========================================
echo 整理完成！
echo ========================================
echo.
echo 核心文档（改善方案/）:
dir /b *.md 2>nul
echo.
echo 归档文档（改善方案/archive/）:
dir /b archive\*.md 2>nul
echo.
echo 建议: 查看 README.md 开始使用
pause
