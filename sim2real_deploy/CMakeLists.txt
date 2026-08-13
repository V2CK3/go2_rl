cmake_minimum_required(VERSION 3.5)
project(sim2real_deploy_lcm)
set(CMAKE_CXX_STANDARD 17)

# ----------------------------------------------------------------------------
# Unitree SDK2（本仓库已放在 bridge/unitree_sdk2）
# 通过 add_subdirectory 导入官方 CMake 目标：unitree_sdk2 / ddsc / ddscxx
# ----------------------------------------------------------------------------
set(UNITREE_SDK2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bridge/unitree_sdk2")
if (NOT EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "unitree_sdk2 not found at ${UNITREE_SDK2_DIR}\n"
        "Please clone/copy Unitree SDK2 into sim2real_deploy/bridge/unitree_sdk2")
endif()

# 不编译 SDK 自带 example，只导入库目标
set(BUILD_EXAMPLES OFF CACHE BOOL "Build unitree_sdk2 examples" FORCE)
add_subdirectory(${UNITREE_SDK2_DIR} ${CMAKE_BINARY_DIR}/unitree_sdk2)

# ----------------------------------------------------------------------------
# LCM（需系统已安装：https://github.com/lcm-proj/lcm）
# ----------------------------------------------------------------------------
find_package(lcm REQUIRED)
if (NOT lcm_FOUND)
    message(FATAL_ERROR "LCM not found. Install lcm then re-run cmake.")
endif()
include_directories(lcm_types)

# ----------------------------------------------------------------------------
# 桥接可执行文件
# ----------------------------------------------------------------------------
# DDS <-> LCM 主程序（依赖 unitree_sdk2）
add_executable(lcm_bridge bridge/lcm_bridge.cpp)
target_link_libraries(lcm_bridge
    unitree_sdk2
    lcm
    rt
    pthread
)

# LCM 通信自检（只订阅打印，不写电机）
add_executable(lcm_receive_msgs_test bridge/lcm_receive_msgs_test.cpp)
target_link_libraries(lcm_receive_msgs_test lcm)

# 运行时需要能找到 thirdparty 里的 libddsc*.so
set(UNITREE_DDS_LIB_DIR "${UNITREE_SDK2_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}")
if (EXISTS "${UNITREE_DDS_LIB_DIR}")
    set_target_properties(lcm_bridge PROPERTIES
        BUILD_RPATH "${UNITREE_DDS_LIB_DIR}"
        INSTALL_RPATH "${UNITREE_DDS_LIB_DIR}"
    )
endif()
