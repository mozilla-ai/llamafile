@echo off
setlocal enabledelayedexpansion
:: Compiles distributable DLL for AMD GPU support (TinyBLAS + ROCm HIP)
:: Parallel version — uses a small C job runner for concurrent compilation
::
:: The artifact will depend on amdhip64.dll, hipblas.dll, and rocblas.dll
:: from the AMD ROCm HIP SDK installation.
::
:: Usage:
::   llamafile\rocm_parallel.bat              Build with TinyBLAS (default)
::   llamafile\rocm_parallel.bat --clean      Clean and rebuild
::   llamafile\rocm_parallel.bat -j8          Use 8 parallel jobs
::   llamafile\rocm_parallel.bat --output X   Specify output path
::
:: Output: ggml-rocm.dll in the repo root (default)

:: -------- directories --------
:: Capture %~dp0 BEFORE any goto (goto corrupts %~dp0 in batch)
for %%I in ("%~dp0.") do set "LLAMAFILE_DIR=%%~fI"
for %%I in ("%~dp0..") do set "REPO_DIR=%%~fI"

:: -------- parse arguments --------
set "CLEAN=0"
set "OUTPUT="
set "JOBS=%NUMBER_OF_PROCESSORS%"
if not defined JOBS set "JOBS=4"

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--clean"  (set "CLEAN=1"     & shift & goto parse_args)
if /i "%~1"=="--output" (set "OUTPUT=%~2"   & shift & shift & goto parse_args)
if /i "%~1"=="--help" (
    echo Usage: rocm_parallel.bat [-jN] [--clean] [--output PATH]
    exit /b 0
)
:: Handle -jN
set "ARG=%~1"
if "!ARG:~0,2!"=="-j" (
    set "JOBS=!ARG:~2!"
    shift
    goto parse_args
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

:: -------- build configuration --------
set "BUILD_DIR=%USERPROFILE%\.cache\llamafile-rocm-build"
if "%OUTPUT%"=="" set "OUTPUT=%REPO_DIR%\ggml-rocm.dll"

:: -------- clean --------
if "%CLEAN%"=="1" (
    if exist "%BUILD_DIR%" (
        echo Cleaning build directory...
        rmdir /s /q "%BUILD_DIR%"
    )
)
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

:: -------- find HIP compiler --------
:: Check HIP_PATH - try auto-detect if not set
if not defined HIP_PATH (
    for /d %%d in ("C:\Program Files\AMD\ROCm\*") do (
        if exist "%%d\bin\clang++.exe" set "HIP_PATH=%%d"
    )
)
if not defined HIP_PATH (
    echo Error: HIP_PATH not set and ROCm not found in default location
    echo Please install AMD ROCm HIP SDK and set HIP_PATH
    exit /b 1
)
set "HIPCC=%HIP_PATH%\bin\clang++.exe"
if not exist "%HIPCC%" (
    echo Error: clang++.exe not found at %HIPCC%
    exit /b 1
)

:: -------- build parallel job runner --------
set "PARALLEL=%BUILD_DIR%\parallel.exe"
if not exist "%PARALLEL%" (
    echo Building parallel job runner...
    cl /nologo /O2 /Fe:"%PARALLEL%" "%LLAMAFILE_DIR%\parallel.c"
    if errorlevel 1 (echo Error: failed to build parallel.exe & exit /b 1)
    :: Clean up .obj left by cl in current directory
    del /q parallel.obj 2>nul
    echo.
)

:: -------- AMD GPU architecture targets --------
:: gfx906:  Vega 20 (Radeon VII, MI50)
:: gfx1030: RDNA2 (RX 6900 XT, RX 6800 series)
:: gfx1031: RDNA2 (RX 6700 series)
:: gfx1032: RDNA2 (RX 6600 series)
:: gfx1100: RDNA3 (RX 7900 XTX, RX 7900 XT)
:: gfx1101: RDNA3 (RX 7800 series)
:: gfx1102: RDNA3 (RX 7600 series)
:: gfx1103: RDNA3 (RX 7000 mobile)
set "ARCH_FLAGS="
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx906"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1030"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1031"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1032"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1100"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1101"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1102"
set "ARCH_FLAGS=%ARCH_FLAGS% --offload-arch=gfx1103"

:: -------- copy TinyBLAS files --------
copy /y "%LLAMAFILE_DIR%\tinyblas.h"       "%BUILD_DIR%\" >nul
copy /y "%LLAMAFILE_DIR%\tinyblas.cu"      "%BUILD_DIR%\" >nul
copy /y "%LLAMAFILE_DIR%\tinyblas-compat.h" "%BUILD_DIR%\" >nul

:: -------- common HIP compiler flags --------
set "COMMON_FLAGS=-x hip -O2 -ffast-math -std=c++17"
set "COMMON_FLAGS=%COMMON_FLAGS% --gpu-max-threads-per-block=1024"
set "COMMON_FLAGS=%COMMON_FLAGS% -Wno-ignored-attributes -Wno-nested-anon-types"
set "COMMON_FLAGS=%COMMON_FLAGS% -I%BUILD_DIR% -I%GGML_INC_DIR% -I%GGML_SRC_DIR% -I%GGML_CUDA_DIR%"
set "COMMON_FLAGS=%COMMON_FLAGS% -I"%HIP_PATH%\include""
set "COMMON_FLAGS=%COMMON_FLAGS% -DNDEBUG -DGGML_BUILD=1 -DGGML_SHARED=1 -DGGML_BACKEND_SHARED=1 -DGGML_BACKEND_BUILD=1 -DGGML_MULTIPLATFORM"
set "COMMON_FLAGS=%COMMON_FLAGS% -DGGML_USE_HIP=1 -DGGML_USE_TINYBLAS=1 -DGGML_HIP_NO_VMM -D__HIP_PLATFORM_AMD__"

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

echo Building ggml-rocm.dll with TinyBLAS...
echo   Version: !GGML_VERSION! (commit: !GGML_COMMIT!)
echo   Source:  %GGML_CUDA_DIR%
echo   Output:  %OUTPUT%
echo   Build:   %BUILD_DIR%
echo   Jobs:    %JOBS%
echo   HIP:     %HIP_PATH%
echo.

:: -------- generate compilation command list --------
set "CMD_FILE=%BUILD_DIR%\compile_cmds.txt"
set "OBJ_FILES="

:: Empty the command file
type nul > "%CMD_FILE%"
set /a CMD_COUNT=0

:: TinyBLAS source
set "OBJ=%BUILD_DIR%\tinyblas.obj"
if not exist "!OBJ!" (
    echo tinyblas.cu:::"%HIPCC%" -c %ARCH_FLAGS% %COMMON_FLAGS% -o "!OBJ!" "%BUILD_DIR%\tinyblas.cu">> "%CMD_FILE%"
    set /a CMD_COUNT+=1
)
set "OBJ_FILES=!OBJ_FILES! "!OBJ!""

:: Main CUDA/HIP sources
for %%f in ("%GGML_CUDA_DIR%\*.cu") do (
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\!BASE!.obj"
    if not exist "!OBJ!" (
        echo !BASE!.cu:::"%HIPCC%" -c %ARCH_FLAGS% %COMMON_FLAGS% -o "!OBJ!" "%%f">> "%CMD_FILE%"
        set /a CMD_COUNT+=1
    )
    set "OBJ_FILES=!OBJ_FILES! "!OBJ!""
)

:: Template-instances sources
for %%f in ("%GGML_CUDA_DIR%\template-instances\*.cu") do (
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\ti-!BASE!.obj"
    if not exist "!OBJ!" (
        echo ti-!BASE!.cu:::"%HIPCC%" -c %ARCH_FLAGS% %COMMON_FLAGS% -o "!OBJ!" "%%f">> "%CMD_FILE%"
        set /a CMD_COUNT+=1
    )
    set "OBJ_FILES=!OBJ_FILES! "!OBJ!""
)

:: -------- compile .cu sources in parallel --------
if !CMD_COUNT! gtr 0 (
    echo Compiling !CMD_COUNT! .cu files with %JOBS% parallel jobs...
    echo.
    "%PARALLEL%" -j%JOBS% "%CMD_FILE%"
    if errorlevel 1 (
        echo.
        echo Error: HIP compilation failed
        exit /b 1
    )
) else (
    echo All .cu files up to date.
)
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
echo Linking ggml-rocm.dll...
:: Collect all .obj files into a response file (command line too long for cmd.exe)
set "LINK_RSP=%BUILD_DIR%\link_objects.rsp"
type nul > "%LINK_RSP%"
for %%f in ("%BUILD_DIR%\*.obj") do (
    set "OBJPATH=%%f"
    echo "!OBJPATH:\=/!">> "%LINK_RSP%"
)
"%HIPCC%" -shared %ARCH_FLAGS% -o "%OUTPUT%" @"%LINK_RSP%" -L"%HIP_PATH%\lib" "%HIP_PATH%\lib\libhipblas.dll.a" "%HIP_PATH%\lib\rocblas.lib" "%HIP_PATH%\lib\amdhip64.lib"
if errorlevel 1 (
    echo Error: linking failed
    exit /b 1
)

echo.
echo Successfully built: %OUTPUT%
for %%f in ("%OUTPUT%") do echo   Size: %%~zf bytes
echo.

endlocal
