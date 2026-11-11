$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$candidates = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'), 'D:\unet_env\Scripts\python.exe', 'D:\U-NET\venv\Scripts\python.exe')
$pythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExe) { $pythonExe = (Get-Command python -ErrorAction Stop).Source }
$demoPort = '8765'
$envPath = Join-Path $PSScriptRoot '.env'
if (Test-Path -LiteralPath $envPath) {
    foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
        if ($line -match '^PORT=(\d+)$') { $demoPort = $Matches[1] }
    }
}
$env:PORT = $demoPort
$demoUrl = 'http://127.0.0.1:' + $demoPort
try {
    $health = Invoke-RestMethod ($demoUrl + '/api/health') -TimeoutSec 2
    if ($health.ok -and $health.model -eq 'UNet') { Start-Process $demoUrl; exit }
} catch {}
$outLog = Join-Path $PSScriptRoot 'server.log'
$errLog = Join-Path $PSScriptRoot 'server-error.log'
Set-Content -LiteralPath $outLog -Value '' -Encoding UTF8
Set-Content -LiteralPath $errLog -Value '' -Encoding UTF8
Start-Process -FilePath $pythonExe -ArgumentList @('server.py') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod ($demoUrl + '/api/health') -TimeoutSec 2
        if ($health.ok -and $health.model -eq 'UNet') { $ready = $true; break }
    } catch {}
}
if (-not $ready) {
    Write-Host 'Demo 后端启动失败。错误日志如下：' -ForegroundColor Red
    if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 30 }
    Read-Host '按回车关闭'
    exit 1
}
Start-Process $demoUrl
Write-Host ('Demo 已启动：' + $demoUrl)
