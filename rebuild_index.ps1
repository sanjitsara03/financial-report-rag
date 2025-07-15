Write-Host "Removing existing database directories"
if (Test-Path "table_db") { Remove-Item -Recurse -Force "table_db" }
if (Test-Path "text_db") { Remove-Item -Recurse -Force "text_db" }

Write-Host "Running chunk_and_index.py"
python chunk_and_index.py

Write-Host "Indexing complete" 