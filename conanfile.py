from conan import ConanFile
from conan.tools.files import copy
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class NCSRecipe(ConanFile):
    name = "norm_constrained_qp_solver"
    version = "0.1.0"

    # Optional metadata
    license = "MIT"
    author = "Ivo Ivanov (ivo.ivanov@tha.de)"
    url = "github.com/iv461/norm_constrained_qp_solver"
    description = "A fast globally optimal solver for norm-constrained non-convex quadratic programs"
    topics = ("optimization")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "test/*"

    def layout(self):
        cmake_layout(self)

    @property
    def _min_cppstd(self):
        return "17"

    def package_id(self):
        # This should be set so that the package id hash does not change based on different build settings, see https://docs.conan.io/2/tutorial/creating_packages/configure_options_settings.html#header-only-libraries
        self.info.clear()
        
    def requirements(self):
        self.requires("eigen/3.4.0", transitive_headers=True)
        self.requires("fmt/10.2.1", transitive_headers=True)
        self.test_requires("gtest/1.14.0")        
    
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
        copy(self, "*.hpp", self.source_folder, self.package_folder)
