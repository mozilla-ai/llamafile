@echo off
setlocal enabledelayedexpansion
:: Compiles distributable DLL for Vulkan GPU support
::
:: Requires the Vulkan SDK (LunarG) with glslc shader compiler.
:: The artifact will depend on vulkan-1.dll (provided by GPU drivers).
::
:: Usage:
::   llamafile\vulkan.bat              Build ggml-vulkan.dll
::   llamafile\vulkan.bat --clean      Clean and rebuild
::   llamafile\vulkan.bat -j8          Use 8 parallel jobs
::   llamafile\vulkan.bat --output X   Specify output path
::
:: Output: ggml-vulkan.dll in the repo root (default)

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
    echo Usage: vulkan.bat [-jN] [--clean] [--output PATH]
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
set "GGML_SRC_DIR=%LLAMA_CPP_DIR%\ggml\src"
set "GGML_INC_DIR=%LLAMA_CPP_DIR%\ggml\include"
set "GGML_VULKAN_DIR=%LLAMA_CPP_DIR%\ggml\src\ggml-vulkan"
set "SHADERS_DIR=%GGML_VULKAN_DIR%\vulkan-shaders"

if not exist "%GGML_VULKAN_DIR%" (
    echo Error: Vulkan source directory not found: %GGML_VULKAN_DIR%
    exit /b 1
)

:: -------- build configuration --------
set "BUILD_DIR=%USERPROFILE%\.cache\llamafile-vulkan-build"
set "SHADERS_BUILD_DIR=%BUILD_DIR%\shaders"
set "SPVDIR=%BUILD_DIR%\spv"
if "%OUTPUT%"=="" set "OUTPUT=%REPO_DIR%\ggml-vulkan.dll"

:: -------- clean --------
if "%CLEAN%"=="1" (
    if exist "%BUILD_DIR%" (
        echo Cleaning build directory...
        rmdir /s /q "%BUILD_DIR%"
    )
)
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%SHADERS_BUILD_DIR%" mkdir "%SHADERS_BUILD_DIR%"
if not exist "%SPVDIR%" mkdir "%SPVDIR%"

:: -------- find Vulkan SDK --------
if not defined VULKAN_SDK (
    for /d %%d in ("C:\VulkanSDK\*") do set "VULKAN_SDK=%%d"
)
if not defined VULKAN_SDK (
    echo Error: VULKAN_SDK not set and Vulkan SDK not found in C:\VulkanSDK\
    echo Please install the Vulkan SDK from https://vulkan.lunarg.com/
    exit /b 1
)

set "GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
if not exist "%GLSLC%" (
    echo Error: glslc.exe not found at %GLSLC%
    echo Please install the Vulkan SDK with shader tools
    exit /b 1
)

if not exist "%VULKAN_SDK%\Lib\vulkan-1.lib" (
    echo Error: vulkan-1.lib not found at %VULKAN_SDK%\Lib\
    exit /b 1
)

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

:: -------- build parallel job runner --------
set "PARALLEL=%BUILD_DIR%\parallel.exe"
if not exist "%PARALLEL%" (
    echo Building parallel job runner...
    cl /nologo /O2 /Fe:"%PARALLEL%" "%LLAMAFILE_DIR%\parallel.c"
    if errorlevel 1 (echo Error: failed to build parallel.exe & exit /b 1)
    del /q parallel.obj 2>nul
    echo.
)

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

echo Building ggml-vulkan.dll...
echo   Version:    !GGML_VERSION! (commit: !GGML_COMMIT!)
echo   Source:     %GGML_VULKAN_DIR%
echo   Output:     %OUTPUT%
echo   Build:      %BUILD_DIR%
echo   Jobs:       %JOBS%
echo   Vulkan SDK: %VULKAN_SDK%
echo.

:: ========================================================================
:: Phase 1: Build vulkan-shaders-gen.exe
:: ========================================================================
echo Phase 1: Building vulkan-shaders-gen...

set "SHADERS_GEN=%BUILD_DIR%\vulkan-shaders-gen.exe"
set "SHADERS_GEN_SRC=%SHADERS_DIR%\vulkan-shaders-gen.cpp"

if not exist "%SHADERS_GEN%" (
    echo   Compiling vulkan-shaders-gen.cpp...
    cl /nologo /EHsc /O2 /std:c++17 /Fe:"%SHADERS_GEN%" "%SHADERS_GEN_SRC%"
    if errorlevel 1 (echo Error: failed to build vulkan-shaders-gen.exe & exit /b 1)
    :: Clean up .obj left by cl
    del /q vulkan-shaders-gen.obj 2>nul
) else (
    echo   vulkan-shaders-gen.exe is up to date
)
echo.

:: ========================================================================
:: Phase 2: Generate shader header
:: ========================================================================
echo Phase 2: Generating shader header...

set "SHADERS_HPP=%BUILD_DIR%\ggml-vulkan-shaders.hpp"
if not exist "%SHADERS_HPP%" (
    echo   Generating ggml-vulkan-shaders.hpp...
    "%SHADERS_GEN%" --output-dir "%SPVDIR%" --target-hpp "%SHADERS_HPP%"
    if errorlevel 1 (echo Error: failed to generate shader header & exit /b 1)
) else (
    echo   Shader header already exists
)
echo.

:: ========================================================================
:: Phase 3: Compile shaders (.comp -> .cpp) in parallel
:: ========================================================================
echo Phase 3: Compiling shaders...

set "SHADER_CMD_FILE=%BUILD_DIR%\shader_cmds.txt"
type nul > "%SHADER_CMD_FILE%"
set /a SHADER_CMD_COUNT=0

for %%f in ("%SHADERS_DIR%\*.comp") do (
    set "SHADER_NAME=%%~nxf"
    set "SHADER_CPP=%SHADERS_BUILD_DIR%\!SHADER_NAME!.cpp"
    if not exist "!SHADER_CPP!" (
        echo !SHADER_NAME!:::"%SHADERS_GEN%" --glslc "%GLSLC%" --source "%%f" --output-dir "%SPVDIR%" --target-hpp "%SHADERS_HPP%" --target-cpp "!SHADER_CPP!">> "%SHADER_CMD_FILE%"
        set /a SHADER_CMD_COUNT+=1
    )
)

if !SHADER_CMD_COUNT! gtr 0 (
    echo   Compiling !SHADER_CMD_COUNT! shaders with %JOBS% parallel jobs...
    "%PARALLEL%" -j%JOBS% "%SHADER_CMD_FILE%"
    if errorlevel 1 (
        echo Error: shader compilation failed
        exit /b 1
    )
) else (
    echo   All shaders up to date.
)
echo.

:: ========================================================================
:: Phase 4: Compile shader .cpp files -> .obj in parallel
:: ========================================================================
echo Phase 4: Compiling shader C++ files...

set "CXX_FLAGS=/c /nologo /EHsc /O2 /GR /MT /std:c++17 /Zc:preprocessor"
set "CXX_FLAGS=%CXX_FLAGS% /I"%GGML_INC_DIR%" /I"%GGML_SRC_DIR%" /I"%BUILD_DIR%""
set "CXX_FLAGS=%CXX_FLAGS% /DNDEBUG /DGGML_BUILD=1 /DGGML_SHARED=1 /DGGML_BACKEND_SHARED=1 /DGGML_BACKEND_BUILD=1 /DGGML_MULTIPLATFORM"

set "CPP_CMD_FILE=%BUILD_DIR%\cpp_cmds.txt"
type nul > "%CPP_CMD_FILE%"
set /a CPP_CMD_COUNT=0

for %%f in ("%SHADERS_BUILD_DIR%\*.cpp") do (
    set "BASE=%%~nf"
    set "OBJ=%BUILD_DIR%\shader-!BASE!.obj"
    if not exist "!OBJ!" (
        echo !BASE!:::cl %CXX_FLAGS% /Fo"!OBJ!" "%%f">> "%CPP_CMD_FILE%"
        set /a CPP_CMD_COUNT+=1
    )
)

if !CPP_CMD_COUNT! gtr 0 (
    echo   Compiling !CPP_CMD_COUNT! shader C++ files with %JOBS% parallel jobs...
    "%PARALLEL%" -j%JOBS% "%CPP_CMD_FILE%"
    if errorlevel 1 (
        echo Error: shader C++ compilation failed
        exit /b 1
    )
) else (
    echo   All shader C++ files up to date.
)
echo.

:: ========================================================================
:: Phase 5: Compile ggml-vulkan.cpp
:: ========================================================================
echo Phase 5: Compiling ggml-vulkan.cpp...

set "VULKAN_OBJ=%BUILD_DIR%\ggml-vulkan.obj"
set "VULKAN_SRC=%GGML_VULKAN_DIR%\ggml-vulkan.cpp"

if not exist "%VULKAN_OBJ%" (
    echo   Compiling ggml-vulkan.cpp...
    cl %CXX_FLAGS% /I"%GGML_VULKAN_DIR%" /I"%VULKAN_SDK%\Include" /Fo"%VULKAN_OBJ%" "%VULKAN_SRC%"
    if errorlevel 1 (echo Error compiling ggml-vulkan.cpp & exit /b 1)
) else (
    echo   ggml-vulkan.obj is up to date
)
echo.

:: ========================================================================
:: Phase 6: Compile core GGML sources
:: ========================================================================
echo Phase 6: Compiling core GGML sources...

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

:: ========================================================================
:: Phase 7: Link ggml-vulkan.dll
:: ========================================================================
echo Phase 7: Linking ggml-vulkan.dll...

:: Collect all .obj files into a response file (command line too long for cmd.exe)
set "LINK_RSP=%BUILD_DIR%\link_objects.rsp"
type nul > "%LINK_RSP%"
for %%f in ("%BUILD_DIR%\*.obj") do echo "%%f">> "%LINK_RSP%"

link /nologo /DLL /OUT:"%OUTPUT%" @"%LINK_RSP%" vulkan-1.lib /LIBPATH:"%VULKAN_SDK%\Lib"
if errorlevel 1 (
    echo Error: linking failed
    exit /b 1
)

echo.
echo Successfully built: %OUTPUT%
for %%f in ("%OUTPUT%") do echo   Size: %%~zf bytes
echo.

endlocal
