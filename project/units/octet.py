import numpy as np

from project.units.buffer import Buffer
from project.units.register_file import RegisterFile

from project.threads.thread import Thread

from project.configs import Config


class Octet:
    def __init__(self, octet_id: int,
                 register_file: RegisterFile,
                 threads: list[Thread]
                 ) -> None:
        self.id: int = octet_id

        self.__register_file: RegisterFile = register_file

        assert len(threads) == Config.num_threads_per_octet
        self.__threads: list[Thread] = threads

        self.__init_modules()

    def __init_modules(self) -> None:
        # BufferA must loads after each STEP and each octet needs 2 BufferA
        self.__first_matrixA_buffer: Buffer = Buffer(
            buffer_size=Config.size_matrixA_buffer
        )
        self.__second_matrixA_buffer: Buffer = Buffer(
            buffer_size=Config.size_matrixA_buffer
        )

        # BufferB must loads after each STEP and each octet needs 1 BufferB
        self.__first_matrixB_buffer: Buffer = Buffer(
            buffer_size=Config.size_matrixB_buffer
        )

        # BuuferC must loads after each clock(pipeline) and each octet needs 2 BufferC
        self.__first_matrixC_pipelined_buffer: Buffer = Buffer(
            buffer_size=Config.size_matrixC_buffer
        )
        self.__second_matrixC_pipelined_buffer: Buffer = Buffer(
            buffer_size=Config.size_matrixC_buffer
        )

    def LD128_octet_matrixA(self, matrix: np.ndarray) -> None:
        for i in range(len(self.__threads)):
            self.__threads[i].LD128_matrixA(
                vectorA=matrix[i, :].tolist()
            )

    def LD128_octet_matrixB(self, matrix: np.ndarray) -> None:
        for i in range(len(self.__threads)):
            self.__threads[i].LD128_matrixB(
                vectorB=matrix[i, :].tolist()
            )

    def LD128_octet_matrixC(self, matrix: np.ndarray) -> None:
        for i in range(len(self.__threads)):
            self.__threads[i].LD128_matrixC(
                vectorC=matrix[i, :].tolist()
            )

    def HMMA884_SET_octet(self, set_number: int) -> None:
        assert set_number >= 0 and set_number < Config.num_sets_per_mma

        for tid in range(len(self.__threads)):
            self.__threads[tid].store_into_matrixA_lane(
                set_number=set_number
            )

            self.__threads[tid].store_into_matrixB_lane(
                set_number=set_number
            )

    def HMMA884_STEP_octet(self, step_number: int) -> None:
        # In the FP16 execution mode we assume that each SET contains 2 STEP
        assert step_number >= 0 or step_number < Config.num_steps_per_mma_set

        self.__first_matrixA_buffer.store_elements(
            elements=self.__threads[0].matrixA_lane.elements +
            self.__threads[1].matrixA_lane.elements +
            self.__threads[2].matrixA_lane.elements +
            self.__threads[3].matrixA_lane.elements
        )

        self.__second_matrixA_buffer.store_elements(
            elements=self.__threads[4].matrixA_lane.elements +
            self.__threads[5].matrixA_lane.elements +
            self.__threads[6].matrixA_lane.elements +
            self.__threads[7].matrixA_lane.elements
        )

        # This condition like the MUX in the architecture
        if step_number == 0:
            self.__first_matrixB_buffer.store_elements(
                elements=self.__threads[0].matrixB_lane.elements +
                self.__threads[1].matrixB_lane.elements +
                self.__threads[2].matrixB_lane.elements +
                self.__threads[3].matrixB_lane.elements
            )
        elif step_number == 1:
            self.__first_matrixB_buffer.store_elements(
                elements=self.__threads[4].matrixB_lane.elements +
                self.__threads[5].matrixB_lane.elements +
                self.__threads[6].matrixB_lane.elements +
                self.__threads[7].matrixB_lane.elements
            )

        for i in range(4):
            self.__threads[i].store_into_matrixC_lane(
                step_number=step_number
            )
            self.__threads[i+4].store_into_matrixC_lane(
                step_number=step_number
            )

            self.__first_matrixC_pipelined_buffer.store_elements(
                elements=self.__threads[i].matrixC_lane.elements
            )
            self.__second_matrixC_pipelined_buffer.store_elements(
                elements=self.__threads[i+4].matrixC_lane.elements
            )

            # j variable change in each clock
            for j in range(Config.num_clocks_each_HMMA884_FP16_FP16):
                # For the first thread group
                self.__threads[j].FDEP_OP(
                    vectorA=self.__first_matrixA_buffer.elements[i*4: i*4 + 4],
                    vectorB=self.__first_matrixB_buffer.elements[j*4: j*4 + 4],
                    valueC=self.__first_matrixC_pipelined_buffer.elements[j]
                )

                # For the second thread group
                self.__threads[j+4].FDEP_OP(
                    vectorA=self.__second_matrixA_buffer.elements[i*4: i*4 + 4],
                    vectorB=self.__first_matrixB_buffer.elements[j*4: j*4 + 4],
                    valueC=self.__second_matrixC_pipelined_buffer.elements[j]
                )

            self.__register_file.store(
                address=self.__threads[i].matrixC_register_addresses[step_number*2],
                elements=[
                    self.__threads[tid].matrixD_lane.elements[0]
                    for tid in range(0, 2)
                ]
            )
            self.__register_file.store(
                address=self.__threads[i].matrixC_register_addresses[step_number*2 + 1],
                elements=[
                    self.__threads[tid].matrixD_lane.elements[0]
                    for tid in range(2, 4)
                ]
            )

            self.__register_file.store(
                address=self.__threads[i + 4].
                matrixC_register_addresses[step_number*2],
                elements=[
                    self.__threads[tid+4].
                    matrixD_lane.elements[0]
                    for tid in range(0, 2)
                ]
            )
            self.__register_file.store(
                address=self.__threads[i + 4].
                matrixC_register_addresses[step_number*2 + 1],
                elements=[
                    self.__threads[tid+4].
                    matrixD_lane.elements[0]
                    for tid in range(2, 4)
                ]
            )

    def ST128_octet_matrixC(self) -> np.ndarray:
        return np.vstack(
            tup=[
                self.__threads[tid].ST128_matrixC()
                for tid in range(len(self.__threads))
            ]
        )
