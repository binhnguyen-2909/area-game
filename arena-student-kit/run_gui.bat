@echo off
setlocal

set ROOT_DIR=%~dp0

call "%ROOT_DIR%compile.bat"
if errorlevel 1 exit /b 1

java -cp "%ROOT_DIR%lib\arena-framework.jar;%ROOT_DIR%out" arenachallenge.Main student.StudentBotImpl
