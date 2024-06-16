class Register:
    def __init__(self) -> None:
        # Each regsiter contains two fp16 elements
        self.__elements: list[float] = [0.0 for _ in range(2)]

    def load(self) -> list[float]:
        # In fact in store, loads 2 of FP16 values
        return self.__elements

    def store(self, elements: list[float]) -> None:
        # In fact in store, loads 2 of FP16 values
        assert len(elements) == 2

        self.__elements = elements
