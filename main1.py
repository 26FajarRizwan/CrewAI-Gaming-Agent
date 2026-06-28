from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import litellm
import os

load_dotenv()


def apply_groq_compat_patch() -> None:
    """Strip CrewAI fields that Groq's API does not accept."""
    original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        messages = kwargs.get("messages")
        if messages:
            kwargs["messages"] = [
                {k: v for k, v in msg.items() if k != "cache_breakpoint"}
                for msg in messages
            ]
        kwargs.pop("is_litellm", None)
        return original_completion(*args, **kwargs)

    litellm.completion = patched_completion


apply_groq_compat_patch()


llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

game_designer_agent = Agent(
    role="Experienced Game Designer",
    goal="Design a comprehensive, engaging, and mechanically sound game based on the user's request.",
    backstory="You are a veteran game designer with 10+ years of expertise in creating hit indie and AAA titles. You excel at narrative design, balancing core gameplay loops, and drafting clear GDDs (Game Design Documents).",
    llm=llm,
    verbose=True # Isko True karne se terminal par agent ki live soch dikhegi
)

user_request = input("Enter your game design request: ")

game_designer_task = Task(
    description=f"Design a game based on the user's request: {user_request}",
    expected_output="A game design document based on the user's request.",
    agent=game_designer_agent,
)
crew = Crew(
    agents=[game_designer_agent],
    tasks=[game_designer_task],
)
result=crew.kickoff()
print(result)