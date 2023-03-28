
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeUserPresets.json)
set(dynaplex_CMakeUserPresets_exists on)
else()
set(dynaplex_CMakeUserPresets_exists off)
endif()

if(${dynaplex_CMakeUserPresets_exists})
message("CMakeUserPresets detected in ${CMAKE_CURRENT_SOURCE_DIR}/CMakeUserPresets.json" )
else()
message("CMakeUserPresets not detected in ${CMAKE_CURRENT_SOURCE_DIR}/CMakeUserPresets.json")
message("Copying default userpresets")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/CMake/resources/CMakeUserPresets.json.in" ${CMAKE_CURRENT_SOURCE_DIR}/CMakeUserPresets.json)
endif()


message("inclusion switches (CMakeUserPresets.json):")
message("dynaplex_enable_pytorch: ${dynaplex_enable_pytorch}")
message("dynaplex_enable_gurobi: ${dynaplex_enable_gurobi}")
message("dynaplex_enable_pythonbindings: ${dynaplex_enable_pythonbindings}")


if(${dynaplex_enable_pytorch})
message("dynaplex_pytorch_path: ${dynaplex_pytorch_path}")
endif()
if(${dynaplex_enable_gurobi})
message("dynaplex_gurobi_path: ${dynaplex_gurobi_path}")
endif()
if(${dynaplex_enable_pythonbindings})
message("dynaplex_pybind_path: ${dynaplex_pybind_path}")
message("dynaplex_python_path : ${dynaplex_python_path}")
endif()



if(${dynaplex_enable_pytorch})	
list(APPEND CMAKE_PREFIX_PATH ${dynaplex_pytorch_path})
find_package(Torch QUIET)
 if(Torch_FOUND)
 message(STATUS "Succesfully found Torch")
else()
 message(STATUS "Torch not found by dynaplex, even though it was requested via dynaplex_enable_pytorch flag. ") 
 message(STATUS "Please specify location of torch in CMAKE_PREFIX_PATH (for your current configuration) in CMakeUserPresets.json (*), or set dynaplex_enable_pytorch to FALSE to completely disable pytorch")
 message(STATUS "CMAKE_PREFIX_PATH:  ${CMAKE_PREFIX_PATH}")
 message(STATUS "(*) this file should be located in the DynaPlex folder, but might be invisible in some IDEs")
 find_package(Torch REQUIRED)
 message(FATAL_ERROR "Torch requested but not provided")
 endif()

else()
 message(STATUS "pytorch DISABLED : you will not be able to use neural networks. ") 
endif()


if(${dynaplex_enable_pythonbindings})
list(APPEND CMAKE_PREFIX_PATH ${dynaplex_pybind_path})
list(APPEND CMAKE_PREFIX_PATH ${dynaplex_python_path})
find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
message(STATUS "Succesfully found pybind11")
else()
 message(STATUS "PyBind11 not found by dynaplex, even though it was requested via dynaplex_enable_pybind flag. ") 
 message(STATUS "Please specify location of pybind11 and python.exe in dynaplex_python_path and dynaplex_pybind_path in CMakeUserPresets.json (*), or set dynaplex_enable_pybind to FALSE to completely disable pytorch")
 message(STATUS "(*) this file should be located in the DynaPlex folder, but might be invisible in some IDEs")
 find_package(pybind11 CONFIG REQUIRED)
endif()

endif()