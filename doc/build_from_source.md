If you want to build the project from source, do the following: 

Tested on:

- Windows
- Ubuntu Linux 22.04 LTS

Thanks to conan, this package is not dependent on the system package manager on Linux, therefore it probabably works on other Linux distros too, just give it a try !

Windows: Do not use the MS-Store Python-version, as this may lead to the conan-command not being installed correctly, instead install Python from python.org


Install conan2: 
```sh
pip install conan~=2.0
```

Do this once:

```sh
conan profile detect
```


Compile and install the pip package with CMake: 

### Linux 

```sh
mkdir Release
conan install . --build=missing --output Release
cd Release
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
```

### Windows
- Install newest Visual Studion Compiler.

```sh
mkdir Release
conan install . --build=missing --output Release
cd Release
cmake .. -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE="conan_toolchain.cmake"
cmake --build . --config Release --target install
```