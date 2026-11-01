@echo off
cd /d "%~dp0"
if not exist "app.py" (
    echo app.py was not found next to run_app.bat.
    echo Extract the whole zip first, then run run_app.bat from the extracted folder.
    pause
    exit /b 1
)
echo Installing requirements (first run only takes a minute)...
python -m pip install --quiet -r requirements.txt
echo Starting the app at http://localhost:8501  (close this window to stop it)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8501"
python -m streamlit run app.py --server.headless true
pause
