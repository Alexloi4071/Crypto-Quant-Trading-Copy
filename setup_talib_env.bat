@echo off
REM TA-Lib 環境設置批次檔
echo 設置 TA-Lib 環境變量...

set TA_LIBRARY_PATH=D:\crypto-quant-trading-copy\ta-lib\lib
set TA_INCLUDE_PATH=D:\crypto-quant-trading-copy\ta-lib\include
set PATH=%PATH%;D:\crypto-quant-trading-copy\ta-lib\bin

echo ✅ TA-Lib 環境變量已設置
echo.
echo 現在可以安裝 TA-Lib:
echo   pip install TA-Lib
echo.
pause
