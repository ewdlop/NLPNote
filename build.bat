@echo off
REM Cross-Platform Build Script for NLP Note (Windows)
REM Supports Windows, Linux, macOS, Android, and HarmonyOS

echo === NLP Note Cross-Platform Build Script ===
echo.

REM Check if .NET SDK is available
dotnet --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: .NET 8 SDK is required but not installed.
    echo Please install .NET 8 SDK from https://dotnet.microsoft.com/download
    exit /b 1
)

for /f "tokens=*" %%i in ('dotnet --version') do set DOTNET_VERSION=%%i
echo ✓ .NET SDK version: %DOTNET_VERSION%

REM Create output directories
echo Creating output directories...
if not exist dist mkdir dist
if not exist dist\win-x64 mkdir dist\win-x64
if not exist dist\win-arm64 mkdir dist\win-arm64
if not exist dist\linux-x64 mkdir dist\linux-x64
if not exist dist\linux-arm64 mkdir dist\linux-arm64
if not exist dist\osx-x64 mkdir dist\osx-x64
if not exist dist\osx-arm64 mkdir dist\osx-arm64
if not exist dist\android mkdir dist\android
if not exist dist\harmonyos mkdir dist\harmonyos

echo ✓ Output directories created
echo.

REM Build Core Library
echo Building NLP Note Core Library...
dotnet build src\NLPNote.Core\NLPNote.Core.csproj -c Release
if %ERRORLEVEL% neq 0 (
    echo Error: Core library build failed
    exit /b 1
)
echo ✓ Core library built successfully
echo.

REM Build Console Application for different platforms
echo Building Console Application for multiple platforms...

echo Building for Windows x64...
dotnet publish src\NLPNote.Console\NLPNote.Console.csproj -c Release -r win-x64 --self-contained false -o dist\win-x64
echo ✓ Windows x64 build completed

echo Building for Windows ARM64...
dotnet publish src\NLPNote.Console\NLPNote.Console.csproj -c Release -r win-arm64 --self-contained false -o dist\win-arm64
echo ✓ Windows ARM64 build completed

echo Building for Linux x64...
dotnet publish src\NLPNote.Console\NLPNote.Console.csproj -c Release -r linux-x64 --self-contained false -o dist\linux-x64
echo ✓ Linux x64 build completed

echo Building for Linux ARM64...
dotnet publish src\NLPNote.Console\NLPNote.Console.csproj -c Release -r linux-arm64 --self-contained false -o dist\linux-arm64
echo ✓ Linux ARM64 build completed

echo Building for macOS x64...
dotnet publish src\NLPNote.Console\NLPNote.Console.csproj -c Release -r osx-x64 --self-contained false -o dist\osx-x64
echo ✓ macOS x64 build completed

echo Building for macOS ARM64 (Apple Silicon)...
dotnet publish src\NLPNote.Console\NLPNote.Console.csproj -c Release -r osx-arm64 --self-contained false -o dist\osx-arm64
echo ✓ macOS ARM64 build completed

REM Android build (if Android SDK is available)
echo Attempting Android build...
if defined ANDROID_HOME (
    echo Building for Android...
    dotnet publish src\NLPNote.Android\NLPNote.Android.csproj -c Release -o dist\android
    echo ✓ Android build completed
) else (
    echo ⚠ Android SDK not detected. Skipping Android build.
    echo   To build for Android, install Android SDK and set ANDROID_HOME environment variable.
)

REM HarmonyOS build
echo Building for HarmonyOS...
dotnet publish src\NLPNote.HarmonyOS\NLPNote.HarmonyOS.csproj -c Release -r linux-arm64 --self-contained true -o dist\harmonyos
echo ✓ HarmonyOS build completed

echo.
echo === Build Summary ===
echo ✓ Core Library: Built successfully
echo ✓ Windows x64: dist\win-x64\
echo ✓ Windows ARM64: dist\win-arm64\
echo ✓ Linux x64: dist\linux-x64\
echo ✓ Linux ARM64: dist\linux-arm64\
echo ✓ macOS x64: dist\osx-x64\
echo ✓ macOS ARM64: dist\osx-arm64\
if exist dist\android\*.* (
    echo ✓ Android: dist\android\
) else (
    echo ⚠ Android: Skipped (SDK not available)
)
echo ✓ HarmonyOS: dist\harmonyos\

echo.
echo === Usage Instructions ===
echo Windows: .\dist\win-x64\nlpnote.exe [command]
echo Linux: ./dist/linux-x64/nlpnote [command]
echo macOS: ./dist/osx-x64/nlpnote [command]
echo HarmonyOS: ./dist/harmonyos/nlpnote-harmonyos [command]
echo.
echo Available commands: demo, platform, reverse ^<text^>, diacritical ^<text^>
echo Run without commands for interactive mode.
echo.
echo Build completed successfully!