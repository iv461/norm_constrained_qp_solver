from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class NCSReceipt(ConanFile):
    name = "norm_constrained_qp_solver"
    version = "1.0"

    # Optional metadata
    license = "MIT"
    author = "Ivo Ivanov (ivo.ivanov@tha.de)"
    url = "github.com/iv461/norm_constrained_qp_solver"
    description = "A globally optimal solver for non-convex norm-constrained quadratic programs"
    topics = ("optimization")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    @property
    def _min_cppstd(self):
        return "17"
        
    def requirements(self):
        self.requires("eigen/3.4.0", transitive_headers=True)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()   

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["hello"]