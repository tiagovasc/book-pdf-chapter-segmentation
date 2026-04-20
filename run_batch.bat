@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Book Chapter Summarizer - Batch Runner (Windows)
REM
REM  Usage:
REM    run_batch.bat                  Process every .pdf inside the "pdfs" folder
REM    run_batch.bat path\to\a.pdf    Process a single PDF
REM    run_batch.bat one.pdf two.pdf  Process each PDF listed on the command line
REM
REM  Output goes to the folder specified by OUTPUT_BASE_DIR (default: .\output).
REM  Each book lands in: output\[BookName_timestamp]\
REM ============================================================

set SCRIPT=%~dp0book_chapter_summarizer.py

if not exist "%SCRIPT%" (
    echo [ERROR] Script not found: %SCRIPT%
    pause
    exit /b 1
)

REM Default output folder lives next to this script.
if "%OUTPUT_BASE_DIR%"=="" set OUTPUT_BASE_DIR=%~dp0output
if not exist "%OUTPUT_BASE_DIR%" mkdir "%OUTPUT_BASE_DIR%"

REM Warn if the API key is missing. The Python script will also check this.
if "%GEMINI_API_KEY%"=="" (
    echo [WARN] GEMINI_API_KEY is not set in the environment.
    echo        Set it with:  set GEMINI_API_KEY=your_key_here
    echo        Or create a .env file next to the script (see .env.example).
    echo.
)

set TOTAL=0
set SUCCESS=0
set FAIL=0

if "%~1"=="" (
    REM No arguments: process everything in .\pdfs
    set INBOX=%~dp0pdfs
    if not exist "!INBOX!" (
        echo [ERROR] No PDF paths given and default inbox not found: !INBOX!
        echo         Create a "pdfs" folder next to this .bat and drop PDFs there,
        echo         or pass PDF paths as arguments.
        pause
        exit /b 1
    )
    for %%F in ("!INBOX!\*.pdf") do (
        set /a TOTAL+=1
        call :process_one "%%~fF" !TOTAL!
    )
) else (
    REM Arguments given: each is a PDF path
    for %%F in (%*) do (
        set /a TOTAL+=1
        call :process_one "%%~fF" !TOTAL!
    )
)

echo.
echo ============================================================
echo   BATCH COMPLETE
echo   Succeeded: !SUCCESS! / !TOTAL!
echo   Failed:    !FAIL! / !TOTAL!
echo   Output:    %OUTPUT_BASE_DIR%
echo ============================================================
echo.
pause
exit /b 0

:process_one
set "PDF=%~1"
set "NUM=%~2"
echo.
echo ============================================================
echo   [!NUM!] Processing: !PDF!
echo ============================================================
echo.
if not exist "!PDF!" (
    echo   [SKIP] File not found: !PDF!
    set /a FAIL+=1
    goto :eof
)
python "%SCRIPT%" "!PDF!"
if !errorlevel! equ 0 (
    echo   [OK] Finished !PDF!
    set /a SUCCESS+=1
) else (
    echo   [FAIL] Exit code !errorlevel! on !PDF!
    set /a FAIL+=1
)
goto :eof
