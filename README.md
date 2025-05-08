# OneMatrix Library

A header-only matrix mathematics library with comprehensive linear algebra operations inspired by the great teacher Professor Gilbert Strang from MIT-18.06 .

## Build System

This project uses Ninja as a build system with Clang as the compiler. The setup is minimal and focused on simplicity for a header-only library.

### Prerequisites
-Basically None but if u need to bat to work then ---------> 
-Os Windows
- Ninja build system (https://ninja-build.org/)
- Clang compiler (https://releases.llvm.org/download.html)

### Building the Project

To build the project:

```bash
# On Windows
build.bat

# Or directly with Ninja
ninja
```

This will compile the test program in `src/main.cpp` that uses the matrix library.

### Running Tests

After building, you can run the tests with:

```bash
# Run with Ninja
ninja run

# Or directly execute the binary
build/matrix_test
```

### Project Structure

- `src/matrix.h` - The header-only matrix library
- `src/main.cpp` - Testing and demonstration code
- `build.ninja` - Ninja build script
- `build.bat` - Windows batch file for building

### Cleaning

To clean the build artifacts:

```bash
ninja clean
```

### using the bat 
# To build only:
```bash
.\build.ps1
```

# To build and run (will keep window open after execution):
```bash
.\build.ps1 -run
```

# To clean build files:
```bash
.\build.ps1 -clean
```

#To run tests:
```bash
.\build.ps1 -test
```

#To run tests and keep window open after execution:
```bash
.\build.ps1 -test -run
```
