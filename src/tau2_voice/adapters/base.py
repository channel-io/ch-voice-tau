from typing import Any
from abc import abstractmethod, ABC
from tau2_voice.models.events import Event


class BaseEventAdapter(ABC):
    @abstractmethod
    def wrap_event(self, event: Event) -> Any:
        """
        Wrap the tau2 event into the format expected by the target system.
        """
        raise NotImplementedError

    @abstractmethod
    def unwrap_event(self, event: Any) -> Event:
        """
        Unwrap the event from the format expected by the tau2 event.
        """
        raise NotImplementedError