from project.threads.thread import Thread

from project.units.register_file import RegisterFile

from project.configs import Config


class Warp:
    def __init__(self, register_file: RegisterFile) -> None:
        self.__threads: list[Thread] = [
            Thread(thread_id=tid, register_file=register_file) for tid in range(Config.num_threads_per_warp)
        ]

    @property
    def threads(self) -> list[Thread]:
        return self.__threads

    def select_by_thread_group_id(self, thread_group_id: int) -> list[Thread]:
        assert thread_group_id >= 0 and thread_group_id < Config.num_thread_groups_per_warp

        return self.__threads[
            thread_group_id * Config.num_threads_per_thread_group: (thread_group_id + 1) * Config.num_threads_per_thread_group
        ]

    def select_by_octet_id(self, octet_id: int) -> list[Thread]:
        assert octet_id >= 0 and octet_id < Config.num_octets_per_warp

        # Octet(i) = TG(i) union TG(i+4)
        return self.select_by_thread_group_id(
            thread_group_id=octet_id
        ) + self.select_by_thread_group_id(
            thread_group_id=octet_id+4
        )

    def reset(self) -> None:
        # reset loaded addresses
        for thread in self.__threads:
            thread.reset()
