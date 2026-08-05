set(LLMM_REQUIRES esp_driver_usb_serial_jtag esp_partition esp_timer)
set(LLMM_DEBUG_C 0)
set(LLMM_DEBUG_ZIG false)

idf_component_register(SRCS "main.c" INCLUDE_DIRS "."
                       REQUIRES ${LLMM_REQUIRES})
target_compile_definitions(${COMPONENT_LIB} PRIVATE LLMM_DEBUG=${LLMM_DEBUG_C})

set(LLMM_SOURCE "${CMAKE_CURRENT_LIST_DIR}/llmm.zig")
set(LLMM_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/llmm.o")
set(LLMM_OPTIONS "${CMAKE_CURRENT_BINARY_DIR}/llmm_options.zig")
file(GENERATE OUTPUT "${LLMM_OPTIONS}" CONTENT "pub const debug = ${LLMM_DEBUG_ZIG};\n")

add_custom_command(
    OUTPUT "${LLMM_OBJECT}"
    COMMAND zig build-obj
            -target riscv32-freestanding -mcpu=generic_rv32+f+m-d -O ReleaseFast
            --dep llmm_options -Mroot=${LLMM_SOURCE}
            -target riscv32-freestanding -mcpu=generic_rv32+f+m-d -O ReleaseFast
            -Mllmm_options=${LLMM_OPTIONS}
            -femit-bin=${LLMM_OBJECT}
    DEPENDS "${LLMM_SOURCE}" "${LLMM_OPTIONS}"
    COMMENT "Compiling Zig LLMM inference kernel for ESP32-P4"
    VERBATIM
)
add_custom_target(zig_llmm ALL DEPENDS "${LLMM_OBJECT}")
add_dependencies(${COMPONENT_LIB} zig_llmm)
target_sources(${COMPONENT_LIB} PRIVATE "${LLMM_OBJECT}")
