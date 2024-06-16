from project.units.register import Register


class Lane:
    def __init__(self, lane_size: int) -> None:
        # lane size in fp16
        self.__lane_size: int = lane_size

        self.__elements: list[float] = [
            0.0 for _ in range(lane_size)
        ]

    @property
    def elements(self) -> list[float]:
        return self.__elements

    def load_into_registers(self) -> list[Register]:
        registers: list[Register] = list()

        for i in range(self.__lane_size//2, 2):
            r = Register()
            r.store(elements=[self.__elements[2*i], self.__elements[2*i+1]])
            registers.append(r)

        return registers

    def store_from_registers(self, registers: list[Register]) -> None:
        assert len(registers) * 2 == self.__lane_size

        for i in range(len(registers)):
            elements: list[float] = registers[i].load()
            self.__elements[i*2] = elements[0]
            self.__elements[i*2+1] = elements[1]

    def store(self, elementns: list[float]) -> None:
        assert len(elementns) == self.__lane_size

        self.__elements = elementns.copy()
