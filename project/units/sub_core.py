import numpy as np

from project.units.register_file import RegisterFile
from project.units.octet import Octet

from project.threads.warp import Warp

from project.configs import Config


class SubCore:
    def __init__(self, sub_core_id: int) -> None:
        self.id: int = sub_core_id
        self.__init_modules()

    def __init_modules(self) -> None:
        self.__register_file: RegisterFile = RegisterFile()

        self.__warp: Warp = Warp(register_file=self.__register_file)

        self.__octets: list[Octet] = list()

        for octet_id in range(Config.num_octets_per_warp):
            self.__octets.append(
                Octet(
                    octet_id=octet_id,
                    register_file=self.__register_file,
                    threads=self.__warp.select_by_octet_id(
                        octet_id=octet_id
                    )
                )
            )

    @property
    def warp(self) -> Warp:
        return self.__warp

    def wmma_load_matrixA(self, matrixA: np.ndarray) -> None:
        # Each wmma.load matrixA breaks into 2 LD.128
        for step in range(2):
            for octet_id in range(4):
                if octet_id == 0 or octet_id == 2:
                    matrix: np.ndarray = matrixA[0:8, step*8: (step+1)*8]

                else:
                    matrix: np.ndarray = matrixA[8:16, step*8: (step+1)*8]

                self.__octets[octet_id].LD128_octet_matrixA(
                    matrix=matrix
                )

    def wmma_load_matrixB(self, matrixB: np.ndarray) -> None:
        # Each wmma.load matrixB breaks into 2 LD.128
        for step in range(2):
            for octet_id in range(4):
                matrix: np.ndarray = None

                if octet_id == 0 or octet_id == 1:
                    matrix = matrixB[0:8, step*8: (step+1)*8]

                else:
                    matrix = matrixB[8:16, step*8: (step+1)*8]

                self.__octets[octet_id].LD128_octet_matrixB(
                    matrix=matrix
                )

    def wmma_load_matrixC(self, matrixC: np.ndarray) -> None:
        for octet_id in range(4):
            if octet_id == 0:
                matrix: np.ndarray = matrixC[0:8, 0:8]

            elif octet_id == 1:
                matrix: np.ndarray = matrixC[8:16, 0:8]

            elif octet_id == 2:
                matrix: np.ndarray = matrixC[0:8, 8:16]

            else:
                matrix: np.ndarray = matrixC[8:16, 8:16]

            self.__octets[octet_id].LD128_octet_matrixC(
                matrix=matrix
            )

    def wmma_mma(self) -> None:
        for octet_id in range(Config.num_octets_per_warp):
            for set_number in range(Config.num_sets_per_mma):
                self.__octets[octet_id].HMMA884_SET_octet(
                    set_number=set_number
                )
                for step_number in range(Config.num_steps_per_mma_set):
                    self.__octets[octet_id].HMMA884_STEP_octet(
                        step_number=step_number
                    )

    def wmma_store(self) -> np.ndarray:
        return np.hstack(
            (
                np.vstack(
                    (
                        self.__octets[0].ST128_octet_matrixC(),
                        self.__octets[1].ST128_octet_matrixC()
                    )
                ),
                np.vstack(
                    (
                        self.__octets[2].ST128_octet_matrixC(),
                        self.__octets[3].ST128_octet_matrixC()
                    )
                )
            )
        )
