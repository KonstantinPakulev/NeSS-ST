from abc import ABC, abstractmethod


class BaseMatcher(ABC):

    @abstractmethod
    def match(self, kp_desc1, kp_desc2):
        ...
