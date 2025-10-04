# Optuna UI 監控腳本
Write-Host "=== Optuna UI 監控開始 ===" -ForegroundColor Green
Write-Host ""

# 1. 檢查 Python 進程
Write-Host "1. 檢查 Python 進程:" -ForegroundColor Cyan
$pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "   ✅ 發現 $($pythonProcesses.Count) 個 Python 進程" -ForegroundColor Green
} else {
    Write-Host "   ❌ 沒有運行的 Python 進程" -ForegroundColor Red
}
Write-Host ""

# 2. 測試主頁面
Write-Host "2. 測試主要頁面:" -ForegroundColor Cyan
$pages = @(
    "http://localhost:8000/",
    "http://localhost:8000/static/optuna.html",
    "http://localhost:8000/static/index.html"
)

foreach ($page in $pages) {
    try {
        $response = Invoke-WebRequest -Uri $page -Method Head -UseBasicParsing -TimeoutSec 3
        $pageName = $page.Split('/')[-1]
        if ($pageName -eq "") { $pageName = "root" }
        Write-Host "   ✅ $pageName : $($response.StatusCode)" -ForegroundColor Green
    } catch {
        $pageName = $page.Split('/')[-1]
        if ($pageName -eq "") { $pageName = "root" }
        Write-Host "   ❌ $pageName : 失敗" -ForegroundColor Red
    }
}
Write-Host ""

# 3. 測試 JavaScript 文件
Write-Host "3. 測試 JavaScript 文件:" -ForegroundColor Cyan
$jsFiles = @(
    "http://localhost:8000/static/js/api.js",
    "http://localhost:8000/static/js/dashboard.js",
    "http://localhost:8000/static/js/realtime.js",
    "http://localhost:8000/static/js/optuna_v2_core.js",
    "http://localhost:8000/static/js/layers/layer0.js",
    "http://localhost:8000/static/js/layers/layer1.js"
)

foreach ($js in $jsFiles) {
    try {
        $response = Invoke-WebRequest -Uri $js -Method Head -UseBasicParsing -TimeoutSec 3
        $fileName = $js.Split('/')[-1]
        Write-Host "   ✅ $fileName : $($response.StatusCode)" -ForegroundColor Green
    } catch {
        $fileName = $js.Split('/')[-1]
        Write-Host "   ❌ $fileName : 失敗 - $($_.Exception.Message)" -ForegroundColor Red
    }
}
Write-Host ""

# 4. 測試 CSS 文件
Write-Host "4. 測試 CSS 文件:" -ForegroundColor Cyan
$cssFiles = @(
    "http://localhost:8000/static/css/optuna_dark.css"
)

foreach ($css in $cssFiles) {
    try {
        $response = Invoke-WebRequest -Uri $css -Method Head -UseBasicParsing -TimeoutSec 3
        $fileName = $css.Split('/')[-1]
        Write-Host "   ✅ $fileName : $($response.StatusCode)" -ForegroundColor Green
    } catch {
        $fileName = $css.Split('/')[-1]
        Write-Host "   ❌ $fileName : 失敗" -ForegroundColor Red
    }
}
Write-Host ""

# 5. 測試 API 端點
Write-Host "5. 測試 API 端點:" -ForegroundColor Cyan
$apiEndpoints = @(
    "http://localhost:8000/api/v1/health",
    "http://localhost:8000/api/v1/portfolio/summary",
    "http://localhost:8000/api/v1/trading/status"
)

foreach ($api in $apiEndpoints) {
    try {
        $response = Invoke-WebRequest -Uri $api -UseBasicParsing -TimeoutSec 3
        $endpointName = $api.Split('/')[-1]
        Write-Host "   ✅ $endpointName : $($response.StatusCode)" -ForegroundColor Green
    } catch {
        $endpointName = $api.Split('/')[-1]
        if ($_.Exception.Response.StatusCode -eq 404) {
            Write-Host "   ⚠️  $endpointName : 404 (未實現，正常)" -ForegroundColor Yellow
        } else {
            Write-Host "   ❌ $endpointName : 失敗" -ForegroundColor Red
        }
    }
}
Write-Host ""

# 6. 檢查端口佔用
Write-Host "6. 檢查端口 8000:" -ForegroundColor Cyan
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($port8000) {
    Write-Host "   ✅ 端口 8000 已被佔用（伺服器運行中）" -ForegroundColor Green
    Write-Host "   進程 ID: $($port8000[0].OwningProcess)" -ForegroundColor Gray
} else {
    Write-Host "   ❌ 端口 8000 未被佔用（伺服器未運行？）" -ForegroundColor Red
}
Write-Host ""

# 7. 總結
Write-Host "=== 監控完成 ===" -ForegroundColor Green
Write-Host ""
Write-Host "訪問 Optuna UI:" -ForegroundColor Cyan
Write-Host "   http://localhost:8000/static/optuna.html" -ForegroundColor White
Write-Host ""
Write-Host "如果發現問題，檢查:" -ForegroundColor Yellow
Write-Host "   1. Python 進程是否在運行" -ForegroundColor Gray
Write-Host "   2. 端口 8000 是否被佔用" -ForegroundColor Gray
Write-Host "   3. 紅色標記的項目" -ForegroundColor Gray

