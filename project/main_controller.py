import numpy as np

from project.units.sub_core import SubCore

from project.utils import Util

from project.configs import Config


class MainController:
    def __init__(self) -> None:
        self.__init_modules()

        self.__create_matrices()

        self.__total_execution_clock: int = 0

        result_with_numpy: np.ndarray = Util.mac_operation(
            matrixA=self.__matrixA,
            matrixB=self.__matrixB,
            matrixC=self.__matrixC
        )

        self.__execute_pipelines()

        result_with_tensor_cores: np.ndarray = self.__matrixC.round(
            decimals=Config.matrix_value_round_precision
        )

        Util.compare_two_matrix(
            first_2d_matrix=result_with_numpy,
            second_2d_matrix=result_with_tensor_cores,
            epsilon=Config.epsilon_difference_with_numpy_result
        )

        print(f'Total execution clock is {self.__total_execution_clock}.')

    def __init_modules(self) -> None:
        # Each SM contains 4 subcores
        self.__sub_cores: list[SubCore] = [
            SubCore(sub_core_id=sub_core_id)
            for sub_core_id in range(Config.num_sub_cores_per_sm)
        ]

    def __create_matrices(self) -> None:
        assert Config.matrixA_dimensions[1] == Config.matrixB_dimensions[0]

        self.__matrixA: np.ndarray = Util.create_random_2d_matrix(
            dimensions=Config.matrixA_dimensions
        )

        self.__matrixB: np.ndarray = Util.create_random_2d_matrix(
            dimensions=Config.matrixB_dimensions
        )

        self.__matrixC: np.ndarray = Util.create_random_2d_matrix(
            dimensions=Config.matrixC_dimensions
        )

    def __execute_pipelines(self) -> None:
        # TODO add padding if the matrix sizes not divided by 16
        sub_core_id: int = 0

        matrixA_list: list[np.ndarray] = list()
        matrixB_list: list[np.ndarray] = list()
        matrixC_list: list[np.ndarray] = list()

        matrixC_indexes: list[tuple[tuple[int], tuple[int]]] = list()

        for k in range(0, Config.matrixB_dimensions[0], 16):
            for i in range(0, Config.matrixA_dimensions[0], 16):
                for j in range(0, Config.matrixB_dimensions[1], 16):
                    matrixA_list.append(self.__matrixA[i:i+16, k:k+16])
                    matrixB_list.append(self.__matrixB[k:k+16, j:j+16].T)
                    matrixC_list.append(self.__matrixC[i:i+16, j:j+16])

                    matrixC_indexes.append(
                        (
                            (i, i+16),
                            (j, j+16)
                        )
                    )

                    sub_core_id += 1

                    if sub_core_id == Config.num_sub_cores_per_sm:
                        self.__execute_sm_wmma(
                            matrixA_list=matrixA_list,
                            matrixB_list=matrixB_list,
                            matrixC_list=matrixC_list
                        )

                        for indexes in range(sub_core_id):
                            self.__matrixC[
                                matrixC_indexes[indexes][0][0]: matrixC_indexes[indexes][0][1],
                                matrixC_indexes[indexes][1][0]: matrixC_indexes[indexes][1][1]
                            ] = matrixC_list[indexes]

                        # reset these variables for next sm execution
                        sub_core_id = 0
                        matrixA_list = list()
                        matrixB_list = list()
                        matrixC_list = list()
                        matrixC_indexes = list()

        # if atleast 1 warp and less than 4 warps exists
        if sub_core_id > 0:
            self.__execute_sm_wmma(
                matrixA_list=matrixA_list,
                matrixB_list=matrixB_list,
                matrixC_list=matrixC_list,
                num_warps=sub_core_id
            )
            for indexes in range(sub_core_id):
                self.__matrixC[
                    matrixC_indexes[indexes][0][0]: matrixC_indexes[indexes][0][1],
                    matrixC_indexes[indexes][1][0]: matrixC_indexes[indexes][1][1]
                ] = matrixC_list[indexes]

    def __execute_sm_wmma(self,
                          matrixA_list: list[np.ndarray],
                          matrixB_list: list[np.ndarray],
                          matrixC_list: list[np.ndarray],
                          num_warps: int = 4
                          ) -> None:
        assert num_warps > 0 and num_warps <= Config.num_sub_cores_per_sm

        for sub_core_id in range(num_warps):
            # reset warp to launch next warp
            self.__sub_cores[sub_core_id].warp.reset()

        for sub_core_id in range(num_warps):
            self.__sub_cores[sub_core_id].wmma_load_matrixA(
                matrixA=matrixA_list[sub_core_id]
            )
        self.__total_execution_clock += Config.num_clocks_wmma_load_matrixA

        for sub_core_id in range(num_warps):
            self.__sub_cores[sub_core_id].wmma_load_matrixB(
                matrixB=matrixB_list[sub_core_id]
            )
        self.__total_execution_clock += Config.num_clocks_wmma_load_matrixB

        for sub_core_id in range(num_warps):
            self.__sub_cores[sub_core_id].wmma_load_matrixC(
                matrixC=matrixC_list[sub_core_id]
            )
        self.__total_execution_clock += Config.num_clocks_wmma_load_matrixC

        for sub_core_id in range(num_warps):
            self.__sub_cores[sub_core_id].wmma_mma()
        self.__total_execution_clock += Config.num_clocks_wmma_mma

        for sub_core_id in range(num_warps):
            matrixC_list[sub_core_id] = self.__sub_cores[sub_core_id].wmma_store()
        self.__total_execution_clock += Config.num_clocks_wmma_store_matrixC
