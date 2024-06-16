from project.units.register import Register

from project.configs import Config


class RegisterFile:
    def __init__(self) -> None:
        self.__register_file_size: int = Config.size_register_file
        self.__registers: list[Register] = [
            Register()
            for _ in range(self.__register_file_size)
        ]

    @property
    def registers(self) -> list[Register]:
        return self.__registers

    def store(self, address: int, elements: list[float]) -> None:
        # Each register equal to 2 * 16bit
        assert len(elements) == 2
        assert address >= 0 and address < self.__register_file_size

        self.__registers[address].store(
            elements=elements
        )

    def load(self, address: int) -> Register:
        assert address >= 0 and address < self.__register_file_size

        return self.__registers[address]
