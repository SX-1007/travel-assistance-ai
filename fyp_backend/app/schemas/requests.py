"""
Request Schemas — Pydantic v2 models for API request validation.

Uses ConfigDict(extra="ignore") for forward compatibility and
str_strip_whitespace=True for automatic input sanitisation.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OnboardingRequest(BaseModel):
    """One-time onboarding profile captured before the app is used.

    Persisted to ``user_profiles`` so the initial trip form no longer has to
    re-collect the user's origin on every submission.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        description="User's display name.",
    )
    origin_country: str = Field(
        ...,
        min_length=1,
        description="User's home country.",
    )
    origin_state: str = Field(
        ...,
        min_length=1,
        description="User's home state or province (mandatory).",
    )


class InitialFormRequest(BaseModel):
    """Initial trip planning form submitted by the user.

    NOTE: ``origin_country`` / ``origin_state`` are NOT collected here — they
    are read from the onboarding profile (``user_profiles``) on the server.
    """

    model_config = ConfigDict(
        extra="ignore",  # Reject unknown fields silently
        str_strip_whitespace=True,  # Auto-strip leading/trailing whitespace
    )

    country: str = Field(
        ...,
        min_length=1,
        description="Destination country.",
    )
    city: List[str] = Field(
        default_factory=list,
        description=(
            "Optional destination states/cities. When omitted, the server "
            "resolves and verifies a planning city before budget assessment."
        ),
    )
    num_people: int = Field(
        ...,
        ge=1,
        description="Number of travellers (mandatory, must be ≥ 1).",
    )
    total_budget: Optional[float] = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Total trip budget in the user's base currency.",
    )
    request_budget_recommendation: bool = Field(
        default=False,
        description=("Explicit opt-in used only when the user does not know a budget."),
    )
    budget_assessment_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Cached assessment identifier echoed only after the user accepts "
            "the recommended minimum."
        ),
    )
    start_date: str = Field(
        ...,
        description="Trip start date in YYYY-MM-DD format.",
    )
    end_date: str = Field(
        ...,
        description="Trip end date in YYYY-MM-DD format.",
    )

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: List[str]) -> List[str]:
        """Allow omission, but reject blank values when localities are supplied."""
        cleaned = [item.strip() for item in value]

        if any(not item for item in cleaned):
            raise ValueError(
                "Every provided destination state/city must be non-empty."
            )

        return cleaned

    @model_validator(mode="after")
    def validate_dates(self) -> "InitialFormRequest":
        """Validate request mode plus ordered real ISO calendar dates."""
        if self.request_budget_recommendation:
            if self.total_budget is not None or self.budget_assessment_id is not None:
                raise ValueError(
                    "Budget recommendation mode cannot include an amount or "
                    "assessment confirmation."
                )
        elif self.total_budget is None:
            raise ValueError(
                "Provide total_budget or explicitly request a budget recommendation."
            )

        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {exc}") from exc

        if end < start:
            raise ValueError("end_date must be on or after start_date")

        return self


class ChatRequest(BaseModel):
    """Chat message request for the conversational agent."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique session identifier for conversation continuity.",
    )
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=8_192,  # Prevent oversized payloads
        description="The user's chat message.",
    )
    budget_action: Literal["accept_recommended"] | None = Field(
        default=None,
        description="Explicit acceptance of the server-owned budget recommendation.",
    )
    budget_assessment_id: str | None = Field(
        default=None,
        min_length=1,
        description="Opaque identifier for the accepted server-owned assessment.",
    )

    @model_validator(mode="after")
    def validate_budget_confirmation(self) -> "ChatRequest":
        """Require the structured confirmation action and opaque ID together."""
        if (self.budget_action is None) != (self.budget_assessment_id is None):
            raise ValueError(
                "budget_action and budget_assessment_id must be provided together"
            )
        return self
