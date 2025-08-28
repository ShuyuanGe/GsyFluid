add_executable(srt_test_main main.cu)
target_link_libraries(srt_test_main velocity_set math collision)
target_compile_options(srt_test_main PUBLIC --expt-relaxed-constexpr)