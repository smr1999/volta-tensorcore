class Buffer:
    def __init__(self, buffer_size: int) -> None:
        self.__buffer_size: int = buffer_size

        self.__elements: list[float] = [0.0 for _ in range(buffer_size)]

    @property
    def elements(self) -> list[float]:
        return self.__elements

    def store_elements(self, elements: list[float]) -> None:
        assert len(elements) == self.__buffer_size

        self.__elements = elements.copy()

    def load_element(self, element_id: int):
        assert element_id >= 0 and element_id < self.__buffer_size

        return self.__elements[element_id]
