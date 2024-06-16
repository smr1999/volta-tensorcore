class ConstantConfig:
    """
        Please don't change these constant configs !!!
    """

    # Each warp consist of 32 threads
    num_threads_per_warp: int = 32

    num_threads_per_thread_group: int = 4
    num_thread_groups_per_warp: int = 8

    num_threads_per_octet: int = 8
    num_octets_per_warp: int = 4

    # Each Subcore has it's own register file
    # Each thread access to 255 register 32bits
    # More info: https://docs.nvidia.com/cuda/ampere-tuning-guide/
    maximum_registers_per_thread: int = 255

    # Suppose each register is 32 bits
    num_registers_operandA_per_thread: int = 8
    num_registers_operandB_per_thread: int = 8
    num_registers_operandC_per_thread: int = 4

    # Size each lane is n * FP16
    size_matrixA_lane: int = 4
    size_matrixB_lane: int = 4
    size_matrixC_lane: int = 4
    size_matrixD_lane: int = 1

    num_sets_per_mma: int = 4
    num_steps_per_mma_set: int = 2

    # Each register is 32 bytes
    # Each sub_core contains a register_file with size 64KB = 16 K Regsiter(size=FP32)
    size_register_file: int = (2**4)*(2**10)

    # size of each buffer means how many FP16 operand can store
    size_matrixA_buffer: int = 16
    size_matrixB_buffer: int = 16
    size_matrixC_buffer: int = 4

    num_clocks_each_HMMA884_FP16_FP16: int = 4

    num_sub_cores_per_sm: int = 4


class ClockSass:
    # TODO add these clock latencies
    LD_E_128: int = 0

    HMMA_E_F16_F16: int = 4

    ST_E_128: int = 0


class Config(ConstantConfig):
    """
        You can freely change these configs !!!
    """

    # make sure that dimensions divide by 16
    # make sure second dimension A = first dimesion B
    # make sure first dimension A = first dimension C
    # make sure second dimension B = second dimension C
    matrixA_dimensions: tuple[int] = (64, 64)
    matrixB_dimensions: tuple[int] = (64, 64)
    matrixC_dimensions: tuple[int] = (64, 64)

    matrix_low_value: float = -50
    matrix_high_value: float = 50
    matrix_value_round_precision: int = 2

    # Change the epsilon may cause a small difference in result
    epsilon_difference_with_numpy_result: float = 0.1

    # Suppose that matrixA in row major
    num_clocks_wmma_load_matrixA: int = 2 * ClockSass.LD_E_128
    # Suppose that matrixB in column major
    num_clocks_wmma_load_matrixB: int = 2 * ClockSass.LD_E_128
    # Suppose that matrixC in row major and FP16 execution mode
    num_clocks_wmma_load_matrixC: int = ClockSass.LD_E_128

    num_clocks_wmma_mma: int = ConstantConfig.num_sets_per_mma * \
        ConstantConfig.num_steps_per_mma_set * \
        ClockSass.HMMA_E_F16_F16

    num_clocks_wmma_store_matrixC: int = ClockSass.ST_E_128
