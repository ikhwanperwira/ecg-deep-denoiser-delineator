@echo off
setlocal

REM Set the URLs and the output file names
set URL1=https://physionet.org/static/published-projects/ludb/lobachevsky-university-electrocardiography-database-1.0.1.zip
set ZIP_FILE1=ludb.zip
set URL2=https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip
set ZIP_FILE2=afdb.zip
set URL3=https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip
set ZIP_FILE3=qtdb.zip
set URL4=https://www.physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip
set ZIP_FILE4=nstdb.zip

REM Download the first file
echo Downloading LUDB dataset...
powershell -Command "Invoke-WebRequest -Uri %URL1% -OutFile %ZIP_FILE1%"

REM Extract the first file
echo Extracting LUDB dataset...
powershell -Command "Expand-Archive -Path %ZIP_FILE1% -DestinationPath ."

REM Delete the first zip file
echo Deleting LUDB zip file...
del %ZIP_FILE1%

REM Download the second file
echo Downloading AFDB dataset...
powershell -Command "Invoke-WebRequest -Uri %URL2% -OutFile %ZIP_FILE2%"

REM Extract the second file
echo Extracting AFDB dataset...
powershell -Command "Expand-Archive -Path %ZIP_FILE2% -DestinationPath ."

REM Delete the second zip file
echo Deleting AFDB zip file...
del %ZIP_FILE2%

REM Download the third file
echo Downloading QTDB dataset...
powershell -Command "Invoke-WebRequest -Uri %URL3% -OutFile %ZIP_FILE3%"

REM Extract the third file
echo Extracting QTDB dataset...
powershell -Command "Expand-Archive -Path %ZIP_FILE3% -DestinationPath ."

REM Delete the third zip file
echo Deleting QTDB zip file...
del %ZIP_FILE3%

REM Download the fourth file
echo Downloading NSTDB dataset...
powershell -Command "Invoke-WebRequest -Uri %URL4% -OutFile %ZIP_FILE4%"

REM Extract the fourth file
echo Extracting NSTDB dataset...
powershell -Command "Expand-Archive -Path %ZIP_FILE4% -DestinationPath ."

REM Delete the fourth zip file
echo Deleting NSTDB zip file...
del %ZIP_FILE4%

echo Done.
endlocal