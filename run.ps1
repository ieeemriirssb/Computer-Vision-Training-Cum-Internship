Set-Location $PSScriptRoot
if (Get-Command py -ErrorAction SilentlyContinue) {
    py main.py
} else {
    python main.py
}
