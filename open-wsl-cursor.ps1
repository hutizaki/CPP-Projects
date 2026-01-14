# Open Cursor in WSL Ubuntu at CPP Projects folder
Set-Location $PSScriptRoot
Start-Process cursor -ArgumentList '--folder-uri', 'vscode-remote://wsl+Ubuntu-24.04/home/hutizaki/git/CPP-Projects'
