# v-askbot PowerShell Launcher
Set-Location -Path $PSScriptRoot
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ' v-askbot: Voice-Enabled Multilingual RAG (HH Goa 2026)' -ForegroundColor Magenta
Write-Host '================================================================' -ForegroundColor Cyan

$oldProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'uvicorn backend.app:app.*--port 8000' }
$oldProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Process python -ArgumentList '-m', 'uvicorn', 'backend.app:app', '--host', '0.0.0.0', '--port', '8000', '--reload'
do {
	Start-Sleep -Milliseconds 500
	try {
		Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 1 | Out-Null
		$ready = $true
	} catch {
		$ready = $false
	}
} until ($ready)

Start-Process 'http://localhost:8000'
Write-Host 'Interface is ready. The server is running in a separate process.' -ForegroundColor Green
