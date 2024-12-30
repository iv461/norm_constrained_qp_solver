If you want to build the project from source, do the following: 

Tested on:

- Windows
- Ubuntu Linux 22.04 LTS

Thanks to conan, this package is not dependent on the system package manager on Linux, therefore it probabably works on other Linux distros too, just give it a try !

A note to Windows Users: Do not use the MS-Store Python-version, as this may lead to the conan-command not being installed correctly, instead install Python from [python.org](https://www.python.org/downloads/)


1. Install conan2: 
```sh
pip install conan~=2.0
```

2. Install a compiler (tested with GCC on Ubuntu and MSVC 17 on Windows)

3. Do this once:

```sh
conan profile detect
```

## Build 

### Linux and Windows

```sh
conan create .
```

## Run unit-tests

### Windows 
```sh
.\build\test\Release\tests.exe
```

### Linux
```sh
.\build\test\tests
```