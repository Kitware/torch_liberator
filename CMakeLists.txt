###
# CMakeLists for TORCH_LIBERATOR C/C++/Cuda Python extention modules
#
# Notes:
# https://github.com/Erotemic/netharn/issues/7
# https://github.com/amueller/word_cloud/pull/42ci

cmake_minimum_required(VERSION 3.13.0)

project(TORCH_LIBERATOR LANGUAGES C CXX)
option(TORCH_LIBERATOR_VERBOSE "Print extra info" FALSE)


# Setup basic python stuff and ensure we have skbuild
###
# Private helper function to execute `python -c "<cmd>"`
#
# Runs a python command and populates an outvar with the result of stdout.
# Be careful of indentation if `cmd` is multiline.
#
function(pycmd outvar cmd)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "${cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  if(NOT ${_exitcode} EQUAL 0)
    message(ERROR "Failed when running python code: \"\"\"
${cmd}\"\"\"")
    message(FATAL_ERROR "Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
endfunction()


###
# Find current python major version user option
#

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})


###
# Find scikit-build and include its cmake resource scripts
#
if (NOT SKBUILD)
  pycmd(skbuild_location "import os, skbuild; print(os.path.dirname(skbuild.__file__))")
  set(skbuild_cmake_dir "${skbuild_location}/resources/cmake")
  # If skbuild is not the driver, then we need to include its utilities in our CMAKE_MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH ${skbuild_cmake_dir})
endif()


find_package(PythonExtensions REQUIRED)



###
# Status string for debugging
#
set(PYTHON_SETUP_STATUS "
  * PYTHON_EXECUTABLE = \"${PYTHON_EXECUTABLE}\"

  * PYTHON_INCLUDE_DIR = \"${PYTHON_INCLUDE_DIR}\"
  * PYTHON_LIBRARY = \"${PYTHON_LIBRARY}\"
  * PYTHON_LIBRARY_DEBUG = \"${PYTHON_LIBRARY_DEBUG}\"

  * skbuild_location = \"${skbuild_location}\"
  * skbuild_cmake_dir = \"${skbuild_cmake_dir}\"
")



find_package(Cython REQUIRED)
#find_package(NumPy REQUIRED)  # not using numpy here


function(cpu_cython_module cython_source module_name)
  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_source}" C OUTPUT_VAR sources)
  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  target_include_directories(
    ${module_name}
    PUBLIC
      #${NumPy_INCLUDE_DIRS}
      ${PYTHON_INCLUDE_DIRS}
  )
  #target_link_libraries(${module_name} ${PYTHON_LIBRARIES})
  #target_link_libraries(${module_name})
  #${PYTHON_LIBRARIES})

  target_compile_definitions(${module_name} PUBLIC
    "NPY_NO_DEPRECATED_API"
    #"NPY_1_7_API_VERSION=0x00000007"
    )

  # Transform the C++ library into an importable python module
  python_extension_module(${module_name})
  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
endfunction(cpu_cython_module)



# Typically you would place this CMake code next to your Cython module
add_subdirectory("torch_liberator/_nx_ext_v2")
#option(BUILD_CPU_FEATURE "A Cython CPU Feature" True)
#if (BUILD_CPU_FEATURE)
#  cpu_cython_module("cython_modename.pyx" "cython_modename")
#endif()



set(TORCH_LIBERATOR_CONFIG_STATUS "
PYTHON_CONFIG_STATUS
====================

Include Dirs:
  * CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES = \"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * PYTHON_INCLUDE_DIRS = \"${PYTHON_INCLUDE_DIRS}\"

status(Cython):
  * CYTHON_FOUND = \"${CYTHON_FOUND}\"
  * CYTHON_EXECUTABLE = \"${CYTHON_EXECUTABLE}\"
  * CYTHON_VERSION = \"${CYTHON_VERSION}\"

status(NumPy):
  * NumPy_FOUND = \"${NumPy_FOUND}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * NumPy_VERSION = \"${NumPy_VERSION}\"
  * NumPy_CONV_TEMPLATE_EXECUTABLE = \"${NumPy_CONV_TEMPLATE_EXECUTABLE}\"
  * NumPy_FROM_TEMPLATE_EXECUTABLE = \"${NumPy_FROM_TEMPLATE_EXECUTABLE}\"

status(PythonExtensions):
  * PYTHON_PREFIX = \"${PYTHON_PREFIX}\"
  * PYTHON_SITE_PACKAGES_DIR = \"${PYTHON_SITE_PACKAGES_DIR}\"
  * PYTHON_RELATIVE_SITE_PACKAGES_DIR = \"${PYTHON_RELATIVE_SITE_PACKAGES_DIR}\"
  * PYTHON_SEPARATOR = \"${PYTHON_SEPARATOR}\"
  * PYTHON_PATH_SEPARATOR = \"${PYTHON_PATH_SEPARATOR}\"
  * PYTHON_EXTENSION_MODULE_SUFFIX = \"${PYTHON_EXTENSION_MODULE_SUFFIX}\"

status(python-setup)
${PYTHON_SETUP_STATUS}
")

if (TORCH_LIBERATOR_VERBOSE)
  message(STATUS ${TORCH_LIBERATOR_CONFIG_STATUS})
endif ()
