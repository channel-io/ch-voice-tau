"""Tool for the user agent to signal task completion."""

from tau2_voice.models.tool import Tool


def stop_conversation() -> str:
    """
    Call this function when your task/issue has been completely resolved and you are satisfied with the outcome.
    
    Use this function when:
    - Your problem has been solved
    - Your request has been fulfilled
    - All your questions have been answered
    - You have nothing more to discuss
    
    Do NOT use this if:
    - You're still waiting for information
    - Your issue is not fully resolved
    - You have follow-up questions
    
    Returns:
        str: Confirmation that the conversation will end
    """
    return "Task completed. Ending conversation."


# Create the tool instance
stop_conversation_tool = Tool(stop_conversation)

