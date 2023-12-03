# For each file in notebooks/, call `nb-clean`

$files = Get-ChildItem -Path notebooks -Recurse -Include *.ipynb
foreach ($file in $files) {
    Write-Host "Cleaning $file"
    nb-clean clean $file
}
