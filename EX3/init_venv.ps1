# Make sure we're in the directory where the file `init_venv.ps1` is located
cd $PSScriptRoot

# Make sure you have Python>=11. Use `Get-Command python` to check
python -m venv .venv

# Source the activate script for Powershell
.\.venv\Scripts\Activate.ps1

# Your prompt should now have a (.venv) prefix. Run `Get-Command pip` to check if you're using the venv pip

# Install our requirements
pip install -r requirements.txt
