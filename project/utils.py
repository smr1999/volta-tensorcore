import numpy as np

from project.configs import Config


class Util:
    @classmethod
    def create_random_2d_matrix(self, dimensions: tuple[int]) -> np.ndarray:
        assert len(dimensions) == 2

        return np.random.uniform(
            low=Config.matrix_low_value,
            high=Config.matrix_high_value,
            size=dimensions
        ).round(decimals=Config.matrix_value_round_precision)

    @classmethod
    def compare_two_matrix(cls, first_2d_matrix: np.ndarray, second_2d_matrix: np.ndarray, epsilon: float = 0.01) -> None:
        assert first_2d_matrix.shape == second_2d_matrix.shape

        for i in range(first_2d_matrix.shape[0]):
            for j in range(first_2d_matrix.shape[1]):
                if np.abs(first_2d_matrix[i, j] - second_2d_matrix[i][j]) > epsilon:
                    print(f'Two matrix is not same at index [{i}][{j}]')
                    return

        print('Two matrix are the same')

    @classmethod
    def mac_operation(self, matrixA: np.ndarray, matrixB: np.ndarray, matrixC: np.ndarray) -> np.ndarray:
        return np.add(
            np.matmul(
                matrixA,
                matrixB
            ),
            matrixC
        ).round(
            decimals=Config.matrix_value_round_precision
        )
