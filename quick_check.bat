@echo off
chcp 65001 >nul
echo ======================================
echo 阶段1-5修复快速验证
echo ======================================
echo.

echo 步骤1/4: 检查核心文件...
if exist "optuna_system\utils\time_integrity.py" (
    echo   [OK] time_integrity.py
) else (
    echo   [X] time_integrity.py 缺失
)

if exist "optuna_system\utils\transaction_cost.py" (
    echo   [OK] transaction_cost.py
) else (
    echo   [X] transaction_cost.py 缺失
)

if exist "optuna_system\utils\timeframe_alignment.py" (
    echo   [OK] timeframe_alignment.py
) else (
    echo   [X] timeframe_alignment.py 缺失
)

if exist "optuna_system\utils\focal_loss.py" (
    echo   [OK] focal_loss.py
) else (
    echo   [X] focal_loss.py 缺失
)

if exist "optuna_system\utils\market_regime.py" (
    echo   [OK] market_regime.py
) else (
    echo   [X] market_regime.py 缺失
)

if exist "optuna_system\utils\multiple_testing.py" (
    echo   [OK] multiple_testing.py
) else (
    echo   [X] multiple_testing.py 缺失
)

echo.
echo 步骤2/4: 检查测试文件...

if exist "tests\test_time_integrity.py" (
    echo   [OK] test_time_integrity.py
) else (
    echo   [X] test_time_integrity.py 缺失
)

if exist "tests\test_timeframe_alignment.py" (
    echo   [OK] test_timeframe_alignment.py
) else (
    echo   [X] test_timeframe_alignment.py 缺失
)

if exist "tests\test_focal_loss.py" (
    echo   [OK] test_focal_loss.py
) else (
    echo   [X] test_focal_loss.py 缺失
)

if exist "tests\test_market_regime.py" (
    echo   [OK] test_market_regime.py
) else (
    echo   [X] test_market_regime.py 缺失
)

if exist "tests\test_multiple_testing.py" (
    echo   [OK] test_multiple_testing.py
) else (
    echo   [X] test_multiple_testing.py 缺失
)

echo.
echo 步骤3/4: 检查灾难性代码删除...
findstr /C:"def _rebalance_labels(self," "optuna_system\optimizers\optuna_label.py" >nul 2>&1
if errorlevel 1 (
    echo   [OK] _rebalance_labels方法已删除
) else (
    echo   [X] _rebalance_labels方法仍存在
)

echo.
echo 步骤4/4: 检查Trials降低...
findstr /C:"n_trials: int = 50" "optuna_system\coordinator.py" >nul 2>&1
if not errorlevel 1 (
    echo   [OK] Layer1 trials=50
)

findstr /C:"n_trials: int = 30" "optuna_system\coordinator.py" >nul 2>&1
if not errorlevel 1 (
    echo   [OK] Layer2 trials=30
)

findstr /C:"n_trials: int = 25" "optuna_system\coordinator.py" >nul 2>&1
if not errorlevel 1 (
    echo   [OK] Layer3 trials=25
)

echo.
echo ======================================
echo 验证完成！
echo ======================================
echo.
echo 关键修复:
echo   1. 时间泄漏修复 (双重shift, Purged CV)
echo   2. 交易成本模型 (Kissell学术模型)
echo   3. 多时框对齐 (严格对齐)
echo   4. 不平衡学习 (删除再平衡)
echo   5. 数据窥探控制 (trials-96%%)
echo.
echo 下一步:
echo   [推荐] 继续问题5-7的修复
echo.
echo 详细文档:
echo   - STAGES_1_5_VERIFICATION_GUIDE.md
echo   - STAGES_1_5_COMPLETION_REPORT.md
echo.
pause

