# GitHub Upload Script - Real Upload from Local Project
$ErrorActionPreference = 'Stop'

$projectRoot = 'D:\crypto-quant-trading-copy1\crypto-quant-trading-copy\crypto-quant-trading-copy\crypto-quant-trading-copy'
Set-Location $projectRoot

Write-Host ""
Write-Host "=== GitHub Upload Process ===" -ForegroundColor Cyan
Write-Host ""

# Configure Git
Write-Host "Step 1: Configuring Git..." -ForegroundColor Yellow
git config user.email "alexloi4071@github.com"
git config user.name "Alexloi4071"
git config core.longpaths true
Write-Host "Git configured." -ForegroundColor Green

# Check current status
Write-Host "`nStep 2: Checking current status..." -ForegroundColor Yellow
$statusOutput = git status --porcelain
$untracked = $statusOutput | Where-Object { $_ -match '^\?\?' }
$modified = $statusOutput | Where-Object { $_ -match '^\s*M' }
$added = $statusOutput | Where-Object { $_ -match '^\s*A' }

Write-Host "Untracked files: $($untracked.Count)" -ForegroundColor Cyan
Write-Host "Modified files: $($modified.Count)" -ForegroundColor Cyan
Write-Host "Staged files: $($added.Count)" -ForegroundColor Cyan

# Add all files (git will respect .gitignore)
Write-Host "`nStep 3: Adding all files (respecting .gitignore)..." -ForegroundColor Yellow
git add -A
Write-Host "Files added to staging." -ForegroundColor Green

# Check what will be committed
Write-Host "`nStep 4: Checking staged files..." -ForegroundColor Yellow
$staged = git diff --cached --name-only | Where-Object { $_ -notmatch '\.md$' }
$stagedMd = git diff --cached --name-only | Where-Object { $_ -match '\.md$' }

Write-Host "Files to commit (excl .md): $($staged.Count)" -ForegroundColor Green
Write-Host "Markdown files to commit: $($stagedMd.Count)" -ForegroundColor Yellow

if ($stagedMd.Count -gt 0) {
    Write-Host "`nRemoving .md files from staging..." -ForegroundColor Yellow
    foreach ($md in $stagedMd) {
        git reset HEAD $md 2>$null
    }
    Write-Host "Markdown files removed from commit." -ForegroundColor Green
}

# Show sample of files to be committed
Write-Host "`nSample files to commit (first 20):" -ForegroundColor Cyan
$staged | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

# Commit
$finalStaged = git diff --cached --name-only
if ($finalStaged) {
    Write-Host "`nStep 5: Creating commit..." -ForegroundColor Yellow
    $commitMsg = "Upload local project files to GitHub (batch sync, excl .md files)"
    git commit -m $commitMsg
    Write-Host "Commit created successfully!" -ForegroundColor Green
    
    # Show commit info
    Write-Host "`nCommit details:" -ForegroundColor Cyan
    git log -1 --stat --oneline
} else {
    Write-Host "`nNo changes to commit." -ForegroundColor Yellow
}

# Check remote status
Write-Host "`nStep 6: Checking remote status..." -ForegroundColor Yellow
git fetch origin
$behind = git rev-list --count HEAD..origin/main
$ahead = git rev-list --count origin/main..HEAD

Write-Host "Local is ahead by: $ahead commits" -ForegroundColor Cyan
Write-Host "Local is behind by: $behind commits" -ForegroundColor Cyan

# Push to GitHub
if ($ahead -gt 0) {
    Write-Host "`nStep 7: Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "Repository: https://github.com/Alexloi4071/Crypto-Quant-Trading-Copy" -ForegroundColor White
    Write-Host "Branch: main" -ForegroundColor White
    Write-Host ""
    
    git push origin main --verbose
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n=== SUCCESS! ===" -ForegroundColor Green
        Write-Host "All files have been pushed to GitHub successfully!" -ForegroundColor Green
    } else {
        Write-Host "`n=== FAILED! ===" -ForegroundColor Red
        Write-Host "Push failed. Please check the error message above." -ForegroundColor Red
    }
} else {
    Write-Host "`nNo commits to push. Already up to date." -ForegroundColor Green
}

Write-Host ""

