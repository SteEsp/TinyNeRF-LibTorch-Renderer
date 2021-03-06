cmake_minimum_required ( VERSION 3.0 )

# Create Project
project ( renderer LANGUAGES CXX ) 

# Find REQUIRED packages
find_package ( OpenCV REQUIRED ) # OpenCV
find_package ( glm CONFIG REQUIRED ) # glm
find_package( Torch REQUIRED )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )

# Add executable
add_executable ( ${PROJECT_NAME} main.cpp )

target_link_libraries ( 
  ${PROJECT_NAME} 
  "${TORCH_LIBRARIES}"
  ${OpenCV_LIBS} 
  glm::glm
)

# Include and Link directories
target_include_directories (
    ${PROJECT_NAME}
    PRIVATE 
      ${OpenCV_INCLUDE_DIRS}
) 

target_link_directories ( 
    ${PROJECT_NAME}
    PRIVATE
      ${OpenCV_LIB_DIR} 
) 

set_property ( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14 )

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:renderer>)
endif (MSVC)