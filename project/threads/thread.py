import numpy as np

from project.units.register_file import RegisterFile
from project.units.fdep import FDEP
from project.units.lane import Lane

from project.configs import Config


class Thread:
    number_of_accessible_registers: int = Config.maximum_registers_per_thread

    def __init__(self, thread_id: int, register_file: RegisterFile) -> None:
        self.id = thread_id
        self.__register_file: RegisterFile = register_file

        self.current_loaded_matrixA_index: int = 0
        self.current_loaded_matrixB_index: int = 0
        self.current_loaded_matrixC_index: int = 0

        self.__matrixA_register_addresses: list[int] = [
            self.id * Thread.number_of_accessible_registers +
            i for i in range(Config.num_registers_operandA_per_thread)
        ]

        self.__matrixB_register_addresses: list[int] = [
            self.id * Thread.number_of_accessible_registers +
            (Config.num_registers_operandA_per_thread) +
            i for i in range(Config.num_registers_operandB_per_thread)
        ]

        self.__matrixC_register_addresses: list[int] = [
            self.id * Thread.number_of_accessible_registers +
            (Config.num_registers_operandA_per_thread + Config.num_registers_operandB_per_thread) +
            i for i in range(Config.num_registers_operandC_per_thread)
        ]

        self.__matrixA_lane: Lane = Lane(
            lane_size=Config.size_matrixA_lane
        )
        self.__matrixB_lane: Lane = Lane(
            lane_size=Config.size_matrixB_lane
        )
        self.__matrixC_lane: Lane = Lane(
            lane_size=Config.size_matrixC_lane
        )
        self.__matrixD_lane: Lane = Lane(
            lane_size=Config.size_matrixD_lane
        )

        self.__fdep: FDEP = FDEP()

    @property
    def matrixA_register_addresses(self) -> list[int]:
        return self.__matrixA_register_addresses

    @property
    def matrixB_register_addresses(self) -> list[int]:
        return self.__matrixB_register_addresses

    @property
    def matrixC_register_addresses(self) -> list[int]:
        return self.__matrixC_register_addresses

    @property
    def matrixA_lane(self) -> Lane:
        return self.__matrixA_lane

    @property
    def matrixB_lane(self) -> Lane:
        return self.__matrixB_lane

    @property
    def matrixC_lane(self) -> Lane:
        return self.__matrixC_lane

    @property
    def matrixD_lane(self) -> Lane:
        return self.__matrixD_lane

    @property
    def fdep(self) -> FDEP:
        return self.__fdep

    def reset(self):
        self.current_loaded_matrixA_index = 0
        self.current_loaded_matrixB_index = 0
        self.current_loaded_matrixC_index = 0

    def LD128_matrixA(self, vectorA: list[float]) -> None:
        for i in range(0, Config.num_registers_operandA_per_thread//2):
            assert self.current_loaded_matrixA_index < len(
                self.__matrixA_register_addresses
            )

            self.__register_file.store(
                address=self.__matrixA_register_addresses[self.current_loaded_matrixA_index],
                elements=[vectorA[i*2], vectorA[i*2+1]]
            )
            self.current_loaded_matrixA_index += 1

    def LD128_matrixB(self, vectorB: list[float]) -> None:
        for i in range(0, Config.num_registers_operandB_per_thread//2):
            assert self.current_loaded_matrixB_index < len(
                self.__matrixB_register_addresses)

            self.__register_file.store(
                address=self.__matrixB_register_addresses[self.current_loaded_matrixB_index],
                elements=[vectorB[i*2], vectorB[i*2+1]]
            )
            self.current_loaded_matrixB_index += 1

    def LD128_matrixC(self, vectorC: list[float]) -> None:
        for i in range(0, Config.num_registers_operandC_per_thread):
            assert self.current_loaded_matrixC_index < len(
                self.__matrixC_register_addresses)

            self.__register_file.store(
                address=self.__matrixC_register_addresses[self.current_loaded_matrixC_index],
                elements=[vectorC[i*2], vectorC[i*2+1]]
            )

            self.current_loaded_matrixC_index += 1

    def store_into_matrixA_lane(self, set_number: int) -> None:
        assert set_number >= 0 and set_number < Config.num_sets_per_mma

        self.__matrixA_lane.store_from_registers(
            registers=[
                self.__register_file.load(
                    address=self.__matrixA_register_addresses[set_number*2]
                ),
                self.__register_file.load(
                    address=self.__matrixA_register_addresses[set_number*2 + 1]
                )
            ]
        )

    def store_into_matrixB_lane(self, set_number: int) -> None:
        assert set_number >= 0 and set_number < Config.num_sets_per_mma

        self.__matrixB_lane.store_from_registers(
            registers=[
                self.__register_file.load(
                    address=self.__matrixB_register_addresses[set_number*2]
                ),
                self.__register_file.load(
                    address=self.__matrixB_register_addresses[set_number*2 + 1]
                )
            ]
        )

    def store_into_matrixC_lane(self, step_number: int) -> None:
        assert step_number >= 0 and step_number < Config.num_steps_per_mma_set

        self.__matrixC_lane.store_from_registers(
            registers=[
                self.__register_file.load(
                    address=self.__matrixC_register_addresses[step_number*2]
                ),
                self.__register_file.load(
                    address=self.__matrixC_register_addresses[step_number*2 + 1]
                )
            ]
        )

    def FDEP_OP(self, vectorA: list[float], vectorB: list[float], valueC: float) -> None:
        assert len(vectorA) == 4
        assert len(vectorB) == 4

        self.__fdep.set_inputs(
            vectorA=vectorA,
            vectorB=vectorB,
            valueC=valueC
        )
        self.__fdep.operate()

        self.__matrixD_lane.store(
            elementns=[self.__fdep.result]
        )

    def ST128_matrixC(self) -> np.ndarray:
        result = []
        for i in range(4):
            result += self.__register_file.load(
                self.__matrixC_register_addresses[i]
            ).load()
        return np.array(
            object=result
        )
