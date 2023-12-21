# For each file in notebooks/, call `nb-clean`

$files = Get-ChildItem -Path notebooks -Recurse -Include *.ipynb
foreach ($file in $files) {
    Write-Host "Cleaning $file"
    python3 C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\nb_clean\__main__.py clean $file
}
