# Mlops
My Mlops Project

## micromamba (local install)

I attempted to install micromamba automatically into this workspace, but the automated download ran into a network/executable error on this machine. To finish installation yourself (recommended), run the helper script `install_micromamba.ps1` in this repository from PowerShell.

Summary of what to do locally (PowerShell):

1. Open PowerShell and change to the repository folder:

```powershell
Set-Location 'C:\Users\Shantanu\Desktop\MLOPS\Mlops'
```

2. Run the bundled installer script (may require changing ExecutionPolicy temporarily):

```powershell
.\install_micromamba.ps1
```

3. If the script succeeds, either initialize your shell with micromamba (recommended) or use `micromamba run` for single commands:

```powershell
# initialize PowerShell for micromamba (you may need to restart the shell)
& .\micromamba.exe shell init -s powershell -p .\micromamba_root

# activate the environment
micromamba activate mlops

# or run a single command without activating
.\micromamba.exe run -n mlops python -c "import sys, numpy as np; print(sys.version.split()[0], np.__version__)"
```

If you prefer, I can try the automated install again if you allow network access or if you'd like me to try a different download mirror.
