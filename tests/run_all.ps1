$ErrorActionPreference = 'Stop'

$Workspace = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Workspace 'fyp_backend'
$Frontend = Join-Path $Workspace 'fyp_frontend'
$Python = Join-Path $Backend 'venv\Scripts\python.exe'

function Assert-CommandSucceeded {
    param([Parameter(Mandatory = $true)][string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Write-Output 'BACKEND: compile'
& $Python -m compileall -q (Join-Path $Backend 'app') (Join-Path $Backend 'main.py') (Join-Path $Backend 'run.py')
Assert-CommandSucceeded 'Backend compilation'

Write-Output 'BACKEND: dependency consistency'
& $Python -m pip check
Assert-CommandSucceeded 'Backend dependency consistency check'

Write-Output 'BACKEND: Ruff'
Push-Location $Backend
try {
    & $Python -m ruff check app main.py run.py ..\tests\backend
    Assert-CommandSucceeded 'Backend Ruff check'
} finally {
    Pop-Location
}

Write-Output 'BACKEND: pytest + coverage'
Push-Location $Backend
try {
    & $Python -m pytest -c ..\tests\pytest.ini ..\tests\backend --cov=app --cov=main --cov-report=term-missing
    Assert-CommandSucceeded 'Backend pytest suite'
} finally {
    Pop-Location
}

Write-Output 'FRONTEND: type check'
Push-Location $Frontend
try {
    & npm.cmd run typecheck
    Assert-CommandSucceeded 'Frontend type check'

    Write-Output 'FRONTEND: Vitest'
    & npm.cmd test
    Assert-CommandSucceeded 'Frontend Vitest suite'

    Write-Output 'FRONTEND: production build'
    & npm.cmd run build
    Assert-CommandSucceeded 'Frontend production build'
} finally {
    Pop-Location
}

Write-Output 'ALL CHECKS PASSED'
