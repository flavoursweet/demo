$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$candidates = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'), 'D:\unet_env\Scripts\python.exe', 'D:\U-NET\venv\Scripts\python.exe')
$pythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExe) { $pythonExe = (Get-Command python -ErrorAction Stop).Source }
& $pythonExe configure_llm.py
if ($LASTEXITCODE -ne 0) { Read-Host 'Press Enter to exit' }
