@echo off
setlocal enabledelayedexpansion
:: Compiles distributable DLL for NVIDIA GPU support (TinyBLAS)
::
:: The artifact will only depend on KERNEL32.DLL and NVCUDA.DLL.
:: NVCUDA DLLs are provided by the installation of the windows GPU
:: driver on a Windows system that has a CUDA-capable GPU installed.
::
:: Usage:
::   llamafile\cuda.bat              Build with TinyBLAS (default)
::   llamafile\cuda.bat --cublas     Build with NVIDIA cuBLAS
::   llamafile\cuda.bat --clean      Clean and rebuild
::
:: Output: ggml-cuda.dll in the repo root (default)

:: -------- directories --------
:: Capture %~dp0 BEFORE any goto (goto corrupts %~dp0 in batch)
for %%I in ("%~dp0.") do set "LLAMAFILE_DIR=%%~fI"
for %%I in ("%~dp0..") do set "REPO_DIR=%%~fI"

:: -------- parse arguments --------
set "USE_CUBLAS=0"
set "CLEAN=0"
set "OUTPUT="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--cublas" (set "USE_CUBLAS=1" & shift & goto parse_args)
if /i "%~1"=="--clean"  (set "CLEAN=1"     & shift & goto parse_args)
if /i "%~1"=="--output" (set "OUTPUT=%~2"   & shift & shift & goto parse_args)
if /i "%~1"=="--help" (
    echo Usage: cuda.bat [--clean] [--cublas] [--output PATH]
    exit /b 0
)
echo Unknown option: %~1
exit /b 1
:done_args

set "LLAMA_CPP_DIR=%REPO_DIR%\llama.cpp"
set "GGML_CUDA_DIR=%LLAMA_CPP_DIR%\ggml\src\ggml-cuda"
set "GGML_SRC_DIR=%LLAMA_CPP_DIR%\ggml\src"
set "GGML_INC_DIR=%LLAMA_CPP_DIR%\ggml\include"

if not exist "%GGML_CUDA_DIR%" (
    echo Error: CUDA source directory not found: %GGML_CUDA_DIR%
    exit /b 1
)

:: -------- BLAS configuration --------
if "%USE_CUBLAS%"=="1" (
    set "BUILD_DIR=%USERPROFILE%\.cache\llamafile-cuda-cublas-build"
    set "BLAS_NAME=cuBLAS"
    set "BLAS_DEFINE=-DGGML_USE_CUBLAS"
    set "LINK_LIBS=-lcuda -lcublas"
) else (
    set "BUILD_DIR=%USERPROFILE%\.cache\llamafile-cuda-build"
    set "BLAS_NAME=TinyBLAS"
    set "BLAS_DEFINE=-DGGML_USE_TINYBLAS"
    set "LINK_LIBS=-lcuda"
)

if "%OUTPUT%"=="" set "OUTPUT=%REPO_DIR%\ggml-cuda.dll"

:: -------- clean --------
if "%CLEAN%"=="1" (
    if exist "%BUILD_DIR%" (
        echo Cleaning build directory...
        rmdir /s /q "%BUILD_DIR%"
    )
)
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

:: -------- check nvcc --------
where nvcc >nul 2>&1
if errorlevel 1 (
    echo Error: nvcc not found in PATH
    echo Please install CUDA toolkit and ensure nvcc is in your PATH
    exit /b 1
)

:: -------- detect CUDA version for Blackwell support --------
set "ARCH_FLAGS="
set "ARCH_FLAGS=%ARCH_FLAGS% -gencode arch=compute_75,code=sm_75"
set "ARCH_FLAGS=%ARCH_FLAGS% -gencode arch=compute_80,code=sm_80"
set "ARCH_FLAGS=%ARCH_FLAGS% -gencode arch=compute_86,code=sm_86"
set "ARCH_FLAGS=%ARCH_FLAGS% -gencode arch=compute_89,code=sm_89"
set "ARCH_FLAGS=%ARCH_FLAGS% -gencode arch=compute_90,code=sm_90"

:: Check for CUDA 13.x to add Blackwell support
for /f "tokens=*" %%v in ('nvcc --version 2^>nul ^| findstr /r "release [0-9]"') do (
    set "NVCC_VER_LINE=%%v"
)
set "CUDA_MAJOR="
if defined NVCC_VER_LINE (
    for /f "tokens=2 delims=," %%a in ("!NVCC_VER_LINE!") do (
        for /f "tokens=2" %%b in ("%%a") do (
            for /f "tokens=1 delims=." %%c in ("%%b") do set "CUDA_MAJOR=%%c"
        )
    )
)
if "%CUDA_MAJOR%"=="13" (
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_120f,code=sm_120f --compress-mode=size"
)

:: -------- copy TinyBLAS files if needed --------
if "%USE_CUBLAS%"=="0" (
    copy /y "%LLAMAFILE_DIR%\tinyblas.h"       "%BUILD_DIR%\" >nul
    copy /y "%LLAMAFILE_DIR%\tinyblas.cu"      "%BUILD_DIR%\" >nul
    copy /y "%LLAMAFILE_DIR%\tinyblas-compat.h" "%BUILD_DIR%\" >nul
)

:: -------- common NVCC flags --------
set "COMMON_FLAGS=--threads 5 --use_fast_math --extended-lambda"
if "%USE_CUBLAS%"=="0" set "COMMON_FLAGS=%COMMON_FLAGS% -I%BUILD_DIR%"
set "COMMON_FLAGS=%COMMON_FLAGS% -I%GGML_INC_DIR% -I%GGML_SRC_DIR% -I%GGML_CUDA_DIR%"
set "COMMON_FLAGS=%COMMON_FLAGS% --forward-unknown-to-host-compiler"
set "COMMON_FLAGS=%COMMON_FLAGS% --std=c++17"
set "COMMON_FLAGS=%COMMON_FLAGS% -Xcompiler="/nologo /EHsc /O2 /GR /MT /std:c++17""
set "COMMON_FLAGS=%COMMON_FLAGS% -DNDEBUG -DGGML_BUILD=1 -DGGML_SHARED=1 -DGGML_BACKEND_SHARED=1 -DGGML_BACKEND_BUILD=1 -DGGML_MULTIPLATFORM"
set "COMMON_FLAGS=%COMMON_FLAGS% %BLAS_DEFINE%"

:: -------- extract GGML version --------
set "GGML_VERSION=unknown"
set "GGML_COMMIT=unknown"
set "CMAKE_FILE=%LLAMA_CPP_DIR%\ggml\CMakeLists.txt"
if exist "%CMAKE_FILE%" (
    for /f "tokens=2 delims=()" %%a in ('findstr /c:"set(GGML_VERSION_MAJOR" "%CMAKE_FILE%"') do (
        for /f "tokens=2" %%v in ("%%a") do set "GGML_VER_MAJOR=%%v"
    )
    for /f "tokens=2 delims=()" %%a in ('findstr /c:"set(GGML_VERSION_MINOR" "%CMAKE_FILE%"') do (
        for /f "tokens=2" %%v in ("%%a") do set "GGML_VER_MINOR=%%v"
    )
    for /f "tokens=2 delims=()" %%a in ('findstr /c:"set(GGML_VERSION_PATCH" "%CMAKE_FILE%"') do (
        for /f "tokens=2" %%v in ("%%a") do set "GGML_VER_PATCH=%%v"
    )
    set "GGML_VERSION=!GGML_VER_MAJOR!.!GGML_VER_MINOR!.!GGML_VER_PATCH!"
)
pushd "%LLAMA_CPP_DIR%\ggml" 2>nul && (
    for /f %%h in ('git rev-parse --short HEAD 2^>nul') do set "GGML_COMMIT=%%h"
    popd
)

echo Building ggml-cuda.dll with %BLAS_NAME%...
echo   Version: !GGML_VERSION! (commit: !GGML_COMMIT!)
echo   Source:  %GGML_CUDA_DIR%
echo   Output:  %OUTPUT%
echo   Build:   %BUILD_DIR%
echo.

:: -------- collect and compile .cu sources --------
set "OBJ_FILES="
set /a COUNT=0
set /a COMPILED=0

:: TinyBLAS source
if "%USE_CUBLAS%"=="0" (
    set /a COUNT+=1
    set "OBJ=%BUILD_DIR%\tinyblas.obj"
    set "SRC=%BUILD_DIR%\tinyblas.cu"
    if not exist "!OBJ!" (
        echo [!COUNT!] Compiling: tinyblas.cu
        nvcc -c %ARCH_FLAGS% %COMMON_FLAGS% -o "!OBJ!" "!SRC!"
        if errorlevel 1 (echo Error compiling tinyblas.cu & exit /b 1)
        set /a COMPILED+=1
    ) else (
        echo [!COUNT!] Skipping: tinyblas.cu (up to date^)
    )
    set "OBJ_FILES=!OBJ_FILES! "!OBJ!""
)

:: Main CUDA sources
for %%f in ("%GGML_CUDA_DIR%\*.cu") do (
    set /a COUNT+=1
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\!BASE!.obj"
    set "SRC=%%f"
    if not exist "!OBJ!" (
        echo [!COUNT!] Compiling: !BASE!.cu
        nvcc -c %ARCH_FLAGS% %COMMON_FLAGS% -o "!OBJ!" "!SRC!"
        if errorlevel 1 (echo Error compiling !BASE!.cu & exit /b 1)
        set /a COMPILED+=1
    ) else (
        echo [!COUNT!] Skipping: !BASE!.cu (up to date^)
    )
    set "OBJ_FILES=!OBJ_FILES! "!OBJ!""
)

:: Template-instances CUDA sources
for %%f in ("%GGML_CUDA_DIR%\template-instances\*.cu") do (
    set /a COUNT+=1
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\ti-!BASE!.obj"
    set "SRC=%%f"
    if not exist "!OBJ!" (
        echo [!COUNT!] Compiling: ti-!BASE!.cu
        nvcc -c %ARCH_FLAGS% %COMMON_FLAGS% -o "!OBJ!" "!SRC!"
        if errorlevel 1 (echo Error compiling ti-!BASE!.cu & exit /b 1)
        set /a COMPILED+=1
    ) else (
        echo [!COUNT!] Skipping: ti-!BASE!.cu (up to date^)
    )
    set "OBJ_FILES=!OBJ_FILES! "!OBJ!""
)

echo.
echo Compiled %COMPILED% of %COUNT% .cu files (rest were up to date)
echo.

:: -------- compile core GGML sources with host compiler --------
echo Compiling core GGML sources...

set "HOST_FLAGS=/nologo /EHsc /O2 /GR /MT /DNDEBUG"
set "HOST_FLAGS=%HOST_FLAGS% /DGGML_BUILD=1 /DGGML_SHARED=1 /DGGML_BACKEND_SHARED=1 /DGGML_BACKEND_BUILD=1 /DGGML_MULTIPLATFORM"
set "HOST_FLAGS=%HOST_FLAGS% /DGGML_VERSION=\"!GGML_VERSION!\" /DGGML_COMMIT=\"!GGML_COMMIT!\""
set "HOST_FLAGS=%HOST_FLAGS% /I"%GGML_INC_DIR%" /I"%GGML_SRC_DIR%""

:: C sources
for %%f in (ggml.c ggml-alloc.c ggml-quants.c) do (
    set "SRC=%GGML_SRC_DIR%\%%f"
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\ggml-core-!BASE!.obj"
    if not exist "!OBJ!" (
        echo   Compiling: %%f
        cl /c %HOST_FLAGS% /Fo"!OBJ!" "!SRC!"
        if errorlevel 1 (echo Error compiling %%f & exit /b 1)
    ) else (
        echo   Skipping: %%f (up to date^)
    )
)

:: C++ sources
for %%f in (ggml-backend.cpp ggml-threading.cpp) do (
    set "SRC=%GGML_SRC_DIR%\%%f"
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\ggml-core-!BASE!.obj"
    if not exist "!OBJ!" (
        echo   Compiling: %%f
        cl /c %HOST_FLAGS% /std:c++17 /Fo"!OBJ!" "!SRC!"
        if errorlevel 1 (echo Error compiling %%f & exit /b 1)
    ) else (
        echo   Skipping: %%f (up to date^)
    )
)

echo.

:: -------- link --------
echo Linking ggml-cuda.dll...
:: Collect all .obj files into a response file (command line too long for cmd.exe)
set "LINK_RSP=%BUILD_DIR%\link_objects.rsp"
type nul > "%LINK_RSP%"
for %%f in ("%BUILD_DIR%\*.obj") do echo "%%f">> "%LINK_RSP%"
nvcc --shared %ARCH_FLAGS% -o "%OUTPUT%" -optf "%LINK_RSP%" %LINK_LIBS%
if errorlevel 1 (
    echo Error: linking failed
    exit /b 1
)

echo.
echo Successfully built: %OUTPUT%
for %%f in ("%OUTPUT%") do echo   Size: %%~zf bytes
echo.

endlocal
