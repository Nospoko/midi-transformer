from abc import abstractmethod


class MidiEncoder:
    @abstractmethod
    def encode(record: dict):
        pass
