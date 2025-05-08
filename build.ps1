# PowerShell build script for the OneMatrix library
param(
    [switch]$clean,
    [switch]$run,
    [switch]$test
)

# Configuration
$buildDir = "build"
$compiler = "g++"
$src = "src/main.cpp"
$testSrc = "tests/matrix_test.cpp"
$output = "$buildDir/algaber.exe"
$testOutput = "$buildDir/matrix_test.exe"

# Check if OpenMP is enabled in library_config.hpp
$configFile = "src/linear_algebra/library_config.hpp"
$openmpEnabled = $false
$threadCount = 2

if (Test-Path $configFile) {
    $configContent = Get-Content $configFile -Raw
    if ($configContent -match "ALGABER_OPENMP_ENABLED\s*=\s*true") {
        $openmpEnabled = $true
        Write-Host "OpenMP is enabled in library_config.hpp" -ForegroundColor Green
        
        # Try to extract thread count
        if ($configContent -match "ThreadCount\s*=\s*(\d+)") {
            $threadCount = [int]$Matches[1]
            Write-Host "Using $threadCount threads for OpenMP" -ForegroundColor Green
        }
    } else {
        Write-Host "OpenMP is disabled in library_config.hpp" -ForegroundColor Yellow
    }
}

# Set compiler flags based on OpenMP configuration
$openmpFlag = if ($openmpEnabled) { "-fopenmp" } else { "" }
$flags = "-Wall -Wextra -std=c++23 -I./src $openmpFlag"
$testFlags = "$flags -I./googletest/googletest/include -I./googletest/googlemock/include"

# Create build directory if it doesn't exist
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
    Write-Host "Created build directory: $buildDir"
}

# Clean build directory if requested
if ($clean) {
    Get-ChildItem -Path $buildDir -Recurse | Remove-Item -Force -Recurse
    Write-Host "Cleaned build directory"
    if (-not $run) { exit 0 }
}

# Function to download Google Test if not already present
function Ensure-GoogleTest {
    $gtestDir = "googletest"
    if (-not (Test-Path $gtestDir)) {
        Write-Host "Downloading Google Test..."
        $tempZip = "$env:TEMP\gtest.zip"
        Invoke-WebRequest -Uri "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip" -OutFile $tempZip
        
        # Extract the zip file
        Write-Host "Extracting Google Test..."
        Expand-Archive -Path $tempZip -DestinationPath $env:TEMP -Force
        
        # Move to the right location
        Move-Item -Path "$env:TEMP\googletest-release-1.12.1" -Destination $gtestDir
        
        # Clean up
        Remove-Item -Path $tempZip -Force
        
        Write-Host "Google Test downloaded and extracted to $gtestDir"
    }
}

# Compile test executable if test flag is used
if ($test) {
    # Ensure Google Test is available
    Ensure-GoogleTest
    
    # Compile Google Test libraries
    Write-Host "Compiling Google Test libraries..."
    $gtestDir = "googletest"
    $gtestSrc = "$gtestDir/googletest/src/gtest-all.cc"
    $gtestMainSrc = "$gtestDir/googletest/src/gtest_main.cc"
    
    # Create lib directory if it doesn't exist
    $libDir = "$buildDir/lib"
    if (-not (Test-Path $libDir)) {
        New-Item -ItemType Directory -Path $libDir | Out-Null
    }
    
    # Compile gtest-all.cc
    $gtestCompileCmd = "$compiler -c $gtestSrc -o $libDir/gtest-all.o $flags -I./googletest/googletest -I./googletest/googletest/include -pthread"
    Write-Host "> $gtestCompileCmd"
    Invoke-Expression $gtestCompileCmd
    
    # Compile gtest_main.cc
    $gtestMainCompileCmd = "$compiler -c $gtestMainSrc -o $libDir/gtest_main.o $flags -I./googletest/googletest -I./googletest/googletest/include -pthread"
    Write-Host "> $gtestMainCompileCmd"
    Invoke-Expression $gtestMainCompileCmd
    
    # Compile test code
    Write-Host "Compiling $testSrc..."
    $testCompileCmd = "$compiler $testFlags $testSrc $libDir/gtest-all.o $libDir/gtest_main.o -o $testOutput -pthread $openmpFlag"
    Write-Host "> $testCompileCmd"
    Invoke-Expression $testCompileCmd
    
    # Run the tests if compilation was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Test compilation successful: $testOutput" -ForegroundColor Green
        
        if ($run) {
            Write-Host "Running tests..." -ForegroundColor Cyan
            & $testOutput
            if ($LASTEXITCODE -eq 0) {
                Write-Host "All tests passed!" -ForegroundColor Green
            } else {
                Write-Host "Some tests failed" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "Test compilation failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    # Regular compilation of the main application
    Write-Host "Compiling $src..."
    $compileCommand = "$compiler $flags $src -o $output"
    Write-Host "> $compileCommand"
    Invoke-Expression $compileCommand
}

# Check if compilation was successful (only for main application, test case is handled above)
if (-not $test -and $LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful: $output" -ForegroundColor Green
    
    # Run the executable if requested
    if ($run) {
        Write-Host "Running $output..." -ForegroundColor Cyan
        Write-Host "Press Enter when done to close this window..." -ForegroundColor Yellow
        & $output
        Read-Host "`nExecution complete! Press Enter to exit"
    }
} elseif (-not $test) {
    Write-Host "Compilation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
