from abc import ABC, abstractmethod
from typing import Literal, AsyncGenerator, Optional

from tau2_voice.models.tool import Tool
from tau2_voice.models.events import Event


class BaseAgent(ABC):
    def __init__(
        self,
        tools: Optional[list[Tool]],
        domain_policy: Optional[str],
        role: Optional[Literal["user", "assistant"]] = None,
        **kwargs,
    ):
        self.tools = tools
        self.domain_policy = domain_policy
        self.role = role
        
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def publish(self, event: Event):
        """
        Publish an outbound message to the transport.
        This is the generalized Pub operation for sending data.
        """
        pass

    @abstractmethod
    async def subscribe(self) -> AsyncGenerator[Event]:
        """
        Subscribe to inbound messages from the transport as an async generator.
        This is the generalized Sub operation for receiving data.
        """
        pass
