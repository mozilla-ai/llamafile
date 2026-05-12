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
set "MINIMAL_ARCHS=0"
set "NO_IQ_QUANTS=0"
set "STRIP=0"
set "COMPRESS=0"
set "FA_ALL_QUANTS=0"

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--cublas"        (set "USE_CUBLAS=1"     & shift & goto parse_args)
if /i "%~1"=="--clean"         (set "CLEAN=1"          & shift & goto parse_args)
if /i "%~1"=="--output"        (set "OUTPUT=%~2"       & shift & shift & goto parse_args)
if /i "%~1"=="--fa-all-quants" (set "FA_ALL_QUANTS=1"  & shift & goto parse_args)
if /i "%~1"=="--minimize-size" (set "MINIMAL_ARCHS=1"  & set "NO_IQ_QUANTS=1" & set "STRIP=1" & set "COMPRESS=1" & shift & goto parse_args)
if /i "%~1"=="--minimal-archs" (set "MINIMAL_ARCHS=1"  & shift & goto parse_args)
if /i "%~1"=="--no-iq-quants"  (set "NO_IQ_QUANTS=1"   & shift & goto parse_args)
if /i "%~1"=="--strip"         (set "STRIP=1"          & shift & goto parse_args)
if /i "%~1"=="--compress"      (set "COMPRESS=1"       & shift & goto parse_args)
if /i "%~1"=="--help" (
    echo Usage: cuda.bat [--clean] [--cublas] [--output PATH]
    echo   --clean          Clean build directory before building
    echo   --cublas         Use NVIDIA cuBLAS instead of TinyBLAS
    echo   --output         Output path for shared library
    echo   --fa-all-quants  Compile all flash-attention vec quant combos
    echo                    ^(default: f16-f16, q4_0-q4_0, q8_0-q8_0, bf16-bf16 only^)
    echo.
    echo Size reduction options ^(all off by default^):
    echo   --minimize-size  Enable all size reduction options below
    echo   --minimal-archs  Use virtual PTX for sm_75/sm_90, real SASS for sm_80/86/89
    echo   --no-iq-quants   Exclude IQ quant MMQ template instances
    echo   --strip          No-op on Windows ^(debug info is in a separate .pdb^)
    echo   --compress       Pass --compress-mode=size to nvcc ^(requires CUDA ^>= 12.8^)
    exit /b 0
)
echo Unknown option: %~1
exit /b 1
:done_args

:: -------- find Visual Studio / Build Tools --------
where cl >nul 2>&1
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo Error: cl.exe not found in PATH and vswhere.exe not found
        echo Please run from a Visual Studio Developer Command Prompt
        exit /b 1
    )
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_PATH=%%i"
    )
    if not defined VS_PATH (
        echo Error: Visual Studio with C++ tools not found
        exit /b 1
    )
    call "!VS_PATH!\VC\Auxiliary\Build\vcvarsall.bat" x64
)

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

:: -------- architecture flags --------
:: Virtual PTX for less-used archs (JIT on first run), real SASS for popular archs.
set "ARCH_FLAGS="
if "%MINIMAL_ARCHS%"=="1" (
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_75,code=compute_75"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_80,code=sm_80"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_86,code=sm_86"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_89,code=sm_89"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_90,code=compute_90"
) else (
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_75,code=sm_75"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_80,code=sm_80"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_86,code=sm_86"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_89,code=sm_89"
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_90,code=sm_90"
)

:: Detect CUDA version for Blackwell and compress support
:: nvcc prints e.g. "Cuda compilation tools, release 13.2, V13.2.51".
:: Splitting on ',', ' ', and '.' produces tokens 4..6 = "release", "13", "2".
set "CUDA_MAJOR="
set "CUDA_MINOR="
for /f "tokens=5,6 delims=, ." %%a in ('nvcc --version 2^>nul ^| findstr /r "release [0-9]"') do (
    set "CUDA_MAJOR=%%a"
    set "CUDA_MINOR=%%b"
)
if "%CUDA_MAJOR%"=="13" (
    set "ARCH_FLAGS=!ARCH_FLAGS! -gencode arch=compute_120f,code=sm_120f"
)

:: Fall back to 0 if version parsing failed, to keep the numeric comparisons below valid.
if not defined CUDA_MAJOR set "CUDA_MAJOR=0"
if not defined CUDA_MINOR set "CUDA_MINOR=0"

:: --compress-mode=size: opt-in via --compress (or --minimize-size). Requires CUDA >= 12.8.
if "%COMPRESS%"=="1" (
    set "COMPRESS_OK=0"
    if !CUDA_MAJOR! gtr 12 set "COMPRESS_OK=1"
    if !CUDA_MAJOR! equ 12 if !CUDA_MINOR! geq 8 set "COMPRESS_OK=1"
    if "!COMPRESS_OK!"=="1" (
        set "ARCH_FLAGS=!ARCH_FLAGS! --compress-mode=size"
    ) else (
        echo Warning: --compress requested but CUDA !CUDA_MAJOR!.!CUDA_MINOR! ^< 12.8; ignoring.
    )
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
set "COMMON_FLAGS=%COMMON_FLAGS% -Xcompiler="/nologo /EHsc /O2 /GR /MT /std:c++17 /Zc:preprocessor""
set "COMMON_FLAGS=%COMMON_FLAGS% -diag-suppress 177 -diag-suppress 221 -diag-suppress 550"
set "COMMON_FLAGS=%COMMON_FLAGS% -DNDEBUG -DGGML_BUILD=1 -DGGML_SHARED=1 -DGGML_BACKEND_SHARED=1 -DGGML_BACKEND_BUILD=1 -DGGML_MULTIPLATFORM"
set "COMMON_FLAGS=%COMMON_FLAGS% %BLAS_DEFINE%"
if "%NO_IQ_QUANTS%"=="1"  set "COMMON_FLAGS=%COMMON_FLAGS% -DGGML_CUDA_NO_IQ_QUANTS"
if "%FA_ALL_QUANTS%"=="1" set "COMMON_FLAGS=%COMMON_FLAGS% -DGGML_CUDA_FA_ALL_QUANTS"

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
if "%MINIMAL_ARCHS%%NO_IQ_QUANTS%%STRIP%%COMPRESS%" neq "0000" (
    echo   Size reduction:
    if "%MINIMAL_ARCHS%"=="1" echo     - Minimal archs ^(PTX for sm_75/sm_90^)
    if "%NO_IQ_QUANTS%"=="1"  echo     - No IQ quant templates
    if "%STRIP%"=="1"         echo     - Strip ^(no-op on Windows; debug info is in a separate .pdb^)
    if "%COMPRESS%"=="1"      echo     - Compress mode enabled
)
if "%FA_ALL_QUANTS%"=="1"     echo   FA all quants: all fattn-vec template instances included
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

:: Template-instances: selectively included to mirror upstream CMake defaults.
set "TI_DIR=%GGML_CUDA_DIR%\template-instances"

:: 2. fattn-mma and fattn-tile instances (always included)
for %%f in ("%TI_DIR%\fattn-mma-*.cu" "%TI_DIR%\fattn-tile-*.cu") do (
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

:: 3. fattn-vec: only the 4 default quant combos unless --fa-all-quants.
if "%FA_ALL_QUANTS%"=="1" (
    set "FATTN_VEC_GLOB=%TI_DIR%\fattn-vec-instance-*.cu"
) else (
    set "FATTN_VEC_GLOB=%TI_DIR%\fattn-vec-instance-f16-f16.cu %TI_DIR%\fattn-vec-instance-q4_0-q4_0.cu %TI_DIR%\fattn-vec-instance-q8_0-q8_0.cu %TI_DIR%\fattn-vec-instance-bf16-bf16.cu"
)
for %%f in (!FATTN_VEC_GLOB!) do (
    if exist "%%f" (
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
)

:: 4. mmf instances (always included)
for %%f in ("%TI_DIR%\mmf-*.cu") do (
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

:: 5. mmq instances: all unless --no-iq-quants excludes mmq-instance-iq*
for %%f in ("%TI_DIR%\mmq-*.cu") do (
    set "BASE=%%~nf"
    set "SKIP=0"
    if "%NO_IQ_QUANTS%"=="1" (
        echo !BASE! | findstr /b /i "mmq-instance-iq" >nul
        if not errorlevel 1 set "SKIP=1"
    )
    if "!SKIP!"=="0" (
        set /a COUNT+=1
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
)

echo.
echo Compiled %COMPILED% of %COUNT% .cu files (rest were up to date)
echo.

:: -------- compile core GGML sources with host compiler --------
echo Compiling core GGML sources...

set "HOST_FLAGS=/nologo /EHsc /O2 /GR /MT /Zc:preprocessor /DNDEBUG"
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
for %%f in (ggml-backend.cpp ggml-backend-meta.cpp ggml-threading.cpp) do (
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

if "%STRIP%"=="1" (
    echo Note: --strip is a no-op on Windows ^(debug info is in a separate .pdb^)
)

echo.
echo Successfully built: %OUTPUT%
for %%f in ("%OUTPUT%") do echo   Size: %%~zf bytes
echo.

endlocal
