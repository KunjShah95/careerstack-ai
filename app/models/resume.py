"""Pydantic schemas for structured resume data.

These are what the LLM extraction step (llm_parse.py) fills in, and what
every scoring component reads from. No logic lives here.
"""

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None


class Experience(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    start_date: str | None = None  # "YYYY-MM", "YYYY", or "Present"
    end_date: str | None = None  # "YYYY-MM", "YYYY", or "Present"
    is_current: bool | None = None
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    start_date: str | None = None  # "YYYY-MM", "YYYY", or "Present"
    end_date: str | None = None  # "YYYY-MM", "YYYY", or "Present"


class Project(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    link: str | None = None


class ParsedResume(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    sections_found: list[str] = Field(default_factory=list)
    total_experience_months: int = 0
