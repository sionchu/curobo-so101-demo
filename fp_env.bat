@echo off
chcp 437 >nul
set VSLANG=1033
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA_HOME=D:\Workspace\curobo-demo\.venv-fp\Lib\site-packages\nvidia\cu13
set CUDA_PATH=D:\Workspace\curobo-demo\.venv-fp\Lib\site-packages\nvidia\cu13
set PATH=D:\Workspace\curobo-demo\.venv-fp\Lib\site-packages\nvidia\cu13\bin;D:\Workspace\curobo-demo\.venv-fp\Lib\site-packages\nvidia\cu13\bin\x64;%PATH%
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=8
%*

