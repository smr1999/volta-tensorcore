
class FP16Operation:
    # Suppose that simulation on MAC operation in FP16 mode
    @classmethod
    def add(cls, first_operand: float, second_operand: float) -> float:
        return first_operand + second_operand

    @classmethod
    def multiply(cls, first_operand: float, second_operand: float) -> float:
        return first_operand * second_operand


class FDEP:
    def __init__(self) -> None:

        self.__vevtorA: list[float] = [0.0 for _ in range(4)]
        self.__vevtorB: list[float] = [0.0 for _ in range(4)]
        self.__valueC: float = 0.0
        self.__result: float = 0.0

    def set_inputs(self, vectorA: list[float], vectorB: list[float], valueC: float) -> None:
        assert len(vectorA) == 4
        assert len(vectorB) == 4

        self.__vevtorA = vectorA
        self.__vevtorB = vectorB
        self.__valueC = valueC
        self.__result = 0.0

    @property
    def result(self) -> float:
        return self.__result

    def operate(self) -> None:
        self.__result = FP16Operation.add(
            first_operand=FP16Operation.add(
                first_operand=FP16Operation.add(
                    first_operand=FP16Operation.multiply(
                        first_operand=self.__vevtorA[0],
                        second_operand=self.__vevtorB[0]
                    ),
                    second_operand=FP16Operation.multiply(
                        first_operand=self.__vevtorA[1],
                        second_operand=self.__vevtorB[1]
                    )
                ),
                second_operand=FP16Operation.add(
                    first_operand=FP16Operation.multiply(
                        first_operand=self.__vevtorA[2],
                        second_operand=self.__vevtorB[2]
                    ),
                    second_operand=FP16Operation.multiply(
                        first_operand=self.__vevtorA[3],
                        second_operand=self.__vevtorB[3]
                    )
                )
            ),
            second_operand=self.__valueC
        )
