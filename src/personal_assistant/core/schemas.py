from datetime import date, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personal_assistant.agent.tools import TOOLS, ToolName


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Action(StrictModel):
    name: ToolName
    title: str = Field(default="", max_length=250)
    content: str = Field(default="", max_length=20000)
    query: str = Field(default="", max_length=2000)
    analysis_id: str = Field(default="", pattern=r"^(?:[a-zA-Z0-9][a-zA-Z0-9_-]{0,63})?$")
    weather_period: Literal["current", "today", "tomorrow", "week", "dates"] = "current"
    hourly: bool = False
    temperature_unit: Literal["celsius", "fahrenheit"] = "celsius"
    resource_id: str = Field(default="", max_length=100)
    url: str = Field(default="", max_length=2048)
    tags: list[str] = Field(default_factory=list, max_length=20)
    due_at: datetime | None = None
    date_from: date | None = None
    date_to: date | None = None
    date_field: Literal["created_at", "updated_at"] = "created_at"
    starts_at: str = Field(
        default="", max_length=60, description="Calendar event local start ISO date/time"
    )
    ends_at: str = Field(
        default="", max_length=60, description="Calendar event local end ISO date/time"
    )
    event_timezone: str = Field(
        default="", max_length=100, description="Calendar event IANA timezone"
    )
    all_day: bool | None = None
    mcp_server: str = ""
    mcp_tool: str = ""
    mcp_arguments: dict = Field(default_factory=dict)

    def validate_arguments(self):
        spec = TOOLS[self.name]
        supplied = self.model_dump(exclude_defaults=True, exclude={"name"})
        if set(supplied) - set(spec.fields.split()):
            raise ValueError("Action contains parameters belonging to another tool")
        for field in spec.required.split():
            value = getattr(self, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"{self.name} requires {field}")


Specialist = Literal["general", "knowledge", "productivity", "research", "career", "analysis"]


class AgentTask(StrictModel):
    specialist: Specialist
    task: str = Field(min_length=1, max_length=1000)
    actions: list[Action] = Field(default_factory=list, max_length=5)


class Plan(StrictModel):
    specialist: Specialist
    actions: list[Action] = Field(default_factory=list, max_length=5)
    needs_review: bool = False
    agents: list[AgentTask] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_team(self):
        actions = self.actions or [action for agent in self.agents for action in agent.actions]
        identities = [
            action.analysis_id
            for action in actions
            if action.name == "run_python" and action.analysis_id
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Analysis tasks in a plan must have distinct analysis_id values")
        if self.agents:
            roles = [agent.specialist for agent in self.agents]
            if self.specialist not in roles:
                raise ValueError("The primary specialist must belong to the team")
            if self.actions or sum(len(agent.actions) for agent in self.agents) > 5:
                raise ValueError("Team plans use agents.actions only, at most five actions total")
            if any(not agent.task.strip() for agent in self.agents):
                raise ValueError("Each agent needs a nonempty task")
            if len(self.agents) == 1:
                # Some local models wrap a single task in team syntax. Normalize
                # that equivalent plan instead of failing a harmless formatting choice.
                self.actions = self.agents[0].actions
                self.agents = []
        return self


class Review(StrictModel):
    decision: Literal["accept", "revise", "need_evidence"]
    issues: list[str] = Field(default_factory=list, max_length=6)


class Summary(StrictModel):
    summary: str = Field(max_length=1800)


class ChatRequest(StrictModel):
    thread_id: str = Field(default="default", min_length=1, max_length=100, pattern=r"^[\w:-]+$")
    request_id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=1, max_length=12000)


class Artifact(StrictModel):
    id: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str = Field(min_length=1, max_length=100)
    media_type: Literal["image/png"] = "image/png"


class ChatResponse(StrictModel):
    request_id: UUID
    thread_id: str
    text: str
    specialist: str
    specialists: list[str] = Field(default_factory=list)
    reviewed: bool
    review_status: str
    review_replans: int = Field(default=0, ge=0, le=5)
    max_review_replans: int = Field(default=2, ge=0, le=5)
    operations: list[dict] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list, max_length=3)
