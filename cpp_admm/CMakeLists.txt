cmake_minimum_required(VERSION 3.16)
project(csvport_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-march=native)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# third-party root built in /tmp/cppadmm (see report_baseline.md)
set(THIRDPARTY "" CACHE PATH "path containing include/ and lib/ of the solvers")

option(USE_OSQP "enable OSQP engine" ON)
option(USE_QPALM "enable QPALM engine" ON)
option(USE_SCS "enable SCS engine" ON)
option(USE_PROXQP "enable ProxQP engine" ON)

add_library(csvport_core STATIC
  src/csv.cpp
  src/model.cpp
  src/proctitle.cpp
  src/engines/engines.cpp
  src/engines/osqp_engine.cpp
  src/engines/qpalm_engine.cpp
  src/engines/scs_engine.cpp
  src/engines/proxqp_engine.cpp
  src/engines/native_engine.cpp
  src/engines/native_qpalm_engine.cpp
  src/engines/native_proxqp_engine.cpp
  src/engines/native_ralm_engine.cpp
)
target_include_directories(csvport_core PRIVATE src src/engines)

if(THIRDPARTY)
  target_include_directories(csvport_core PRIVATE ${THIRDPARTY}/include)
endif()

if(USE_OSQP)
  target_compile_definitions(csvport_core PRIVATE USE_OSQP)
  # link a symbol-renamed static osqp (osqpv1_ prefix, see report_baseline.md
  # build notes): qpalm exports C symbols with identical names
  # (check_termination, scale_data, ...) and dynamic resolution would hijack
  # them into the exe's static copy -> memory corruption.
  if(THIRDPARTY AND EXISTS /tmp/cppadmm/osqp_renamed/libosqpr.a)
    target_link_libraries(csvport_core PRIVATE /tmp/cppadmm/osqp_renamed/libosqpr.a)
  elseif(THIRDPARTY AND EXISTS ${THIRDPARTY}/lib/libosqpstatic.a)
    target_link_libraries(csvport_core PRIVATE ${THIRDPARTY}/lib/libosqpstatic.a)
  else()
    target_link_libraries(csvport_core PRIVATE osqp)
  endif()
endif()
if(USE_QPALM)
  target_compile_definitions(csvport_core PRIVATE USE_QPALM)
  # qpalm needs ladel + blas/lapack + lapacke (libtmglib.so.3 pulled in via full path)
  target_link_libraries(csvport_core PRIVATE qpalm ladel lapack blas lapacke)
endif()
if(USE_SCS)
  target_compile_definitions(csvport_core PRIVATE USE_SCS)
  # scs shared: static scs and static osqp both define the global `oact`
  # (sigaction scratch), so at most one of them may be static in a binary.
  target_link_libraries(csvport_core PRIVATE scsdir m)
endif()
if(USE_PROXQP)
  target_compile_definitions(csvport_core PRIVATE USE_PROXQP)
endif()

add_executable(csvport_cpp src/main.cpp)
target_link_libraries(csvport_cpp PRIVATE csvport_core)
target_include_directories(csvport_cpp PRIVATE src src/engines)

add_executable(rolling_osqp test/rolling_osqp.cpp)
target_link_libraries(rolling_osqp PRIVATE csvport_core)
target_include_directories(rolling_osqp PRIVATE src src/engines)
if(THIRDPARTY)
  target_include_directories(rolling_osqp PRIVATE ${THIRDPARTY}/include)
  add_executable(iter_check test/iter_check.cpp)
  add_executable(iter_check2 test/iter_check2.cpp)
  add_executable(nqp_check test/nqp_check.cpp)
  add_executable(nqp_unit test/nqp_unit.cpp)
  target_link_libraries(nqp_unit PRIVATE csvport_core)
  target_include_directories(nqp_unit PRIVATE src src/engines ${THIRDPARTY}/include)
  target_link_directories(nqp_unit PRIVATE ${THIRDPARTY}/lib)
  if(EXISTS ${THIRDPARTY}/lapacke_root)
    target_link_directories(nqp_unit PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu)
    target_link_libraries(nqp_unit PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu/libtmglib.so.3)
    set_target_properties(nqp_unit PROPERTIES BUILD_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu;${THIRDPARTY}/lib")
  endif()
  target_link_libraries(nqp_check PRIVATE csvport_core)
  target_include_directories(nqp_check PRIVATE src src/engines ${THIRDPARTY}/include)
  target_link_directories(nqp_check PRIVATE ${THIRDPARTY}/lib)
  if(EXISTS ${THIRDPARTY}/lapacke_root)
    target_link_directories(nqp_check PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu)
    target_link_libraries(nqp_check PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu/libtmglib.so.3)
    set_target_properties(nqp_check PROPERTIES BUILD_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu;${THIRDPARTY}/lib")
  endif()
  target_link_libraries(iter_check2 PRIVATE csvport_core)
  target_include_directories(iter_check2 PRIVATE src src/engines ${THIRDPARTY}/include)
  target_link_directories(iter_check2 PRIVATE ${THIRDPARTY}/lib)
  if(EXISTS ${THIRDPARTY}/lapacke_root)
    target_link_directories(iter_check2 PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu)
    target_link_libraries(iter_check2 PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu/libtmglib.so.3)
    set_target_properties(iter_check2 PROPERTIES BUILD_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu;${THIRDPARTY}/lib")
  endif()
  target_link_libraries(iter_check PRIVATE csvport_core)
  target_include_directories(iter_check PRIVATE src src/engines ${THIRDPARTY}/include)
  target_link_directories(iter_check PRIVATE ${THIRDPARTY}/lib)
  if(EXISTS ${THIRDPARTY}/lapacke_root)
    target_link_directories(iter_check PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu)
    target_link_libraries(iter_check PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu/libtmglib.so.3)
    set_target_properties(iter_check PROPERTIES BUILD_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu;${THIRDPARTY}/lib")
  endif()
  target_link_directories(rolling_osqp PRIVATE ${THIRDPARTY}/lib)
  if(EXISTS ${THIRDPARTY}/lapacke_root)
    target_link_directories(rolling_osqp PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu)
    target_link_libraries(rolling_osqp PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu/libtmglib.so.3)
    set_target_properties(rolling_osqp PROPERTIES BUILD_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu;${THIRDPARTY}/lib")
  endif()
endif()
if(THIRDPARTY)
  target_include_directories(csvport_cpp PRIVATE ${THIRDPARTY}/include)
  target_link_directories(csvport_cpp PRIVATE ${THIRDPARTY}/lib)
  if(EXISTS ${THIRDPARTY}/lapacke_root)
    target_link_directories(csvport_cpp PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu)
    target_link_libraries(csvport_cpp PRIVATE ${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu/libtmglib.so.3)
    set_target_properties(csvport_cpp PROPERTIES BUILD_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu"
      INSTALL_RPATH "${THIRDPARTY}/lapacke_root/usr/lib/x86_64-linux-gnu;${THIRDPARTY}/lib")
  endif()
endif()
