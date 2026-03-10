Development / setup (Windows PowerShell)
=====================================

Quick checklist to eliminate the 7 Pylance warnings in `backend/train_model.py` (scikit-learn, pandas, joblib imports):

1) Create & activate the venv (if you haven't already)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
# optional extras we use
pip install joblib numpy
```

3) Make sure VS Code uses the workspace venv

- Command Palette -> Python: Select Interpreter -> choose the interpreter that starts with
  `C:/Users/user/Documents/Women_Safety_Project/.venv/Scripts/python.exe`
- Then: Developer: Reload Window

4) Quick import check

```powershell
C:/Users/user/Documents/Women_Safety_Project/.venv/Scripts/python.exe -c "import pandas, sklearn, joblib, numpy; print('imports ok')"
```

5) If Pylance still reports "could not be resolved from source" for binary packages

- This repo workspace now includes a VS Code setting that suppresses that diagnostic (it is noisy, not a runtime error):
  - `.vscode/settings.json` contains `python.analysis.diagnosticSeverityOverrides` set to silence `reportMissingModuleSource`.
- If you'd rather not silence it, remove that override and ensure the correct interpreter is selected and packages are installed in the selected venv.

Notes
- The warnings you saw are either (A) missing packages from the selected Python interpreter or (B) Pylance reporting 'missing module source' for compiled wheels (scikit-learn, numpy). The steps above cover both cases.

If you want, I can also:
- Add a short `README.md` with the same steps.
- Run a quick Problems refresh for you and report back (I cannot reload your VS Code window from here). Provide a screenshot after you reload and I'll iterate until clean.
