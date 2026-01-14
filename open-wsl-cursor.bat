@echo off
REM Open Cursor in WSL Ubuntu at CPP Projects folder
cd /d "%~dp0"
start "" cursor --folder-uri "vscode-remote://wsl+Ubuntu-24.04/home/hutizaki/git/CPP-Projects"
