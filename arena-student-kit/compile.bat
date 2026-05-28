@echo off
setlocal

set ROOT_DIR=%~dp0
if not exist "%ROOT_DIR%out" mkdir "%ROOT_DIR%out"

javac --release 17 ^
  -cp "%ROOT_DIR%lib\arena-framework.jar" ^
  -d "%ROOT_DIR%out" ^
  "%ROOT_DIR%src\student\StudentBotImpl.java"

if errorlevel 1 exit /b 1

echo Compiled StudentBotImpl.java
