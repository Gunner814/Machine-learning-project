@echo off
:: This file is designed to run from the Build folder
:: it will try clone dependencies online if the folder doesnt exist (It assume it is there if there is a folder with the same name)

SET CURR_DIR=%cd%

SET RED=[91m
SET GREEN=[92m
SET YELLOW=[93m
SET BLUE=[94m
::No Color
SET NC=[0m

if NOT exist "dependencies" mkdir dependencies

::Git----------------------------------------------------------------------------------------------------------
if exist "C:/Program Files/Git" goto GIT_DONE
cd dependencies
echo %YELLOW%=================================================================
echo Git not found in default path:(C:/Program Files/Git)
echo Downloading Git 64-bit... (Cancel download if already owned)
echo Installation options can be left as Default

powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe', 'Git-2.44.0-64-bit.exe')"
Git-2.36.1-64-bit.exe
::go back the starting path
cd %CURR_DIR%
if NOT exist "C:/Program Files/CMake" goto CMAKE_DOWNLOAD
echo %BLUE%Rerun the bat file after download.%NC%
goto EXIT

:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::librosa----------------------------------------------------------------------------------------------------------
if exist dependencies/librosa goto librosa_DONE
echo %YELLOW%=================================================================
echo librosa not found. 
echo Downloading librosa...%NC%
rmdir "dependencies/librosa" /S /Q
git clone https://github.com/librosa/librosa.git "dependencies/librosa"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%librosa downloaded

:librosa_DONE
::::-------------------------------------------------------------------------------------------------------------

:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::numpy----------------------------------------------------------------------------------------------------------
if exist dependencies/numpy goto numpy_DONE
echo %YELLOW%=================================================================
echo numpy not found. 
echo Downloading numpy...%NC%
rmdir "dependencies/numpy" /S /Q
git clone https://github.com/numpy/numpy.git "dependencies/numpy"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%numpy downloaded

:numpy_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::matplotlib----------------------------------------------------------------------------------------------------------
if exist dependencies/matplotlib goto matplotlib_DONE
echo %YELLOW%=================================================================
echo matplotlib not found. 
echo Downloading matplotlib...%NC%
rmdir "dependencies/matplotlib" /S /Q
git clone https://github.com/matplotlib/matplotlib.git "dependencies/matplotlib"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%matplotlib downloaded

:matplotlib_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::pandas----------------------------------------------------------------------------------------------------------
if exist dependencies/pandas goto pandas_DONE
echo %YELLOW%=================================================================
echo pandas not found. 
echo Downloading pandas...%NC%
rmdir "dependencies/pandas" /S /Q
git clone https://github.com/pandas-dev/pandas.git "dependencies/pandas"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%pandas downloaded

:pandas_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::ipython----------------------------------------------------------------------------------------------------------
if exist dependencies/ipython goto ipython_DONE
echo %YELLOW%=================================================================
echo ipython not found. 
echo Downloading ipython...%NC%
rmdir "dependencies/ipython" /S /Q
git clone https://github.com/ipython/ipython.git "dependencies/ipython"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%ipython downloaded

:ipython_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::plotly----------------------------------------------------------------------------------------------------------
if exist dependencies/plotly goto plotly_DONE
echo %YELLOW%=================================================================
echo plotly not found. 
echo Downloading plotly...%NC%
rmdir "dependencies/plotly" /S /Q
git clone https://github.com/plotly/plotly.git "dependencies/plotly"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%plotly downloaded

:plotly_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::seaborn----------------------------------------------------------------------------------------------------------
if exist dependencies/seaborn goto seaborn_DONE
echo %YELLOW%=================================================================
echo seaborn not found. 
echo Downloading seaborn...%NC%
rmdir "dependencies/seaborn" /S /Q
git clone https://github.com/mwaskom/seaborn.git "dependencies/seaborn"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%seaborn downloaded

:seaborn_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::keras----------------------------------------------------------------------------------------------------------
if exist dependencies/keras goto keras_DONE
echo %YELLOW%=================================================================
echo keras not found. 
echo Downloading keras...%NC%
rmdir "dependencies/keras" /S /Q
git clone https://github.com/keras-team/keras.git "dependencies/keras"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%keras downloaded

:keras_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


::scikit-learn----------------------------------------------------------------------------------------------------------
if exist dependencies/scikit-learn goto scikit-learn_DONE
echo %YELLOW%=================================================================
echo scikit-learn not found. 
echo Downloading scikit-learn...%NC%
rmdir "dependencies/scikit-learn" /S /Q
git clone https://github.com/scikit-learn/scikit-learn.git "dependencies/scikit-learn"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%scikit-learn downloaded

:scikit-learn_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------

::joblib----------------------------------------------------------------------------------------------------------
if exist dependencies/joblib goto joblib_DONE
echo %YELLOW%=================================================================
echo joblib not found. 
echo Downloading joblib...%NC%
rmdir "dependencies/joblib" /S /Q
git clone https://github.com/joblib/joblib.git "dependencies/joblib"
if %ERRORLEVEL% GEQ 1 goto :ERROR
echo %GREEN%joblib downloaded

:joblib_DONE
::::-------------------------------------------------------------------------------------------------------------
:GIT_DONE
::-------------------------------------------------------------------------------------------------------------


:END
echo %GREEN%--------------------
echo %GREEN%Everything is set up
echo %GREEN%--------------------
echo %GREEN%Restart Visual Studios if it does not work%NC%
echo %RED%If you want to redownload any of the dependencies, please delete that folder before rerunning the bat file%NC%
goto EXIT

:ERROR
echo %RED%ERROR!!%NC%
pause

:EXIT
pause