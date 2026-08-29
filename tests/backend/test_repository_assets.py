"""Integrity tests for project-owned reference, configuration and data assets."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
BACKEND = WORKSPACE / "fyp_backend"


def _env_values(relative_path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (
        (WORKSPACE / relative_path).read_text(encoding="utf-8").splitlines()
    ):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name] = value
    return values


def _load_json(relative_path: str):
    return json.loads((WORKSPACE / relative_path).read_text(encoding="utf-8-sig"))


def _private_reference(relative_path: str) -> Path:
    """Return a private reference path or skip when it is absent from a clone."""
    path = WORKSPACE / relative_path
    if not path.exists():
        pytest.skip(f"optional private reference is unavailable: {relative_path}")
    return path


@pytest.mark.parametrize(
    "relative_path",
    [
        "fyp_backend/Malaysia_Embassy_Complete.json",
        "fyp_frontend/package.json",
        "fyp_frontend/package-lock.json",
    ],
)
def test_versioned_json_assets_are_parseable(relative_path: str):
    assert _load_json(relative_path) is not None


def test_root_gitignore_protects_private_and_generated_assets():
    patterns = set((WORKSPACE / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {
        "/extra_info/",
        ".env",
        ".env.*",
        "**/FIREBASE_CREDENTIAL_JSON.json",
        "__pycache__/",
        "venv/",
        "node_modules/",
        "dist/",
    }.issubset(patterns)


def test_root_gitignore_keeps_required_document_and_data_formats_visible():
    patterns = set((WORKSPACE / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert "*.json" not in patterns
    assert "*.md" not in patterns


def test_backend_manual_guide_allows_reconciled_canonical_budget_subsets():
    guide = (WORKSPACE / "docs/testing/backend-manual-testing.md").read_text(
        encoding="utf-8"
    )
    assert "may use a non-empty subset" in guide
    assert "rather than requiring all six keys" in guide
    assert "Its six-category allocation is illustrative" in guide
    assert "allocation contains exactly" not in guide
    assert "currency and six-category allocation" not in guide


def test_backend_manual_guide_documents_strict_location_map_and_budget_evidence():
    guide = (WORKSPACE / "docs/testing/backend-manual-testing.md").read_text(
        encoding="utf-8"
    )
    initial_success = guide.split("### 4.1 Initial-form success", 1)[1].split(
        "### 4.2 Initial planning unavailable", 1
    )[0]
    chat_success = guide.split("### 4.4 Chat success and chat planning unavailable", 1)[
        1
    ].split("### 4.5 History accepted snapshot", 1)[0]
    history_success = guide.split("### 4.5 History accepted snapshot", 1)[1].split(
        "## 5. Manual scenarios", 1
    )[0]
    for example in (initial_success, chat_success, history_success):
        assert '"requested_city": "Singapore"' in example
        assert '"verified_locality": "Singapore"' in example
        assert '"latitude": 1.29' in example
        assert '"longitude": 103.85' in example
    assert "strict route/map parity" in guide
    budget_gate = guide.split("### 4.3 Budget gate responses", 1)[1].split(
        "### 4.4 Chat success and chat planning unavailable", 1
    )[0]
    assert '"hotel_nights": 0' in budget_gate
    assert '"hotel_price_per_night": 0' in budget_gate
    assert "both must be zero or both must be positive" in budget_gate
    assert '"requested_city": "Singapore",\n          "latitude"' not in guide


def test_backend_environment_example_documents_required_settings():
    values = _env_values("fyp_backend/.env.example")
    required = {
        "GEMINI_API_KEY",
        "GEMINI_IATA_API",
        "GEMINI_FLIGHTS_HOTELS",
        "GEMINI_EXTRACTOR_API",
        "GEMINI_EMBED_API",
        "SUPABASE_URL",
        "SUPABASE_API_KEY",
        "SUPABASE_POSTGRES_URI",
        "FIREBASE_CREDENTIAL_JSON",
        "MAPBOX_TOKEN",
        "SERPAPI_FLIGHT",
        "SERPAPI_HOTEL",
        "SERPAPI_PLACE",
        "FREECURRENCY_API",
    }
    assert required.issubset(values)


def test_frontend_environment_example_contains_no_mapbox_credential():
    values = _env_values("fyp_frontend/.env.example")
    assert values["VITE_API_BASE_URL"]
    assert values["VITE_MAPBOX_TOKEN"] == ""


@pytest.mark.parametrize(
    "relative_path",
    [
        "extra_info/Malaysia_Embassy_Complete.json",
        "extra_info/malaysian_missions_complete.json",
    ],
)
def test_private_reference_json_assets_are_parseable_when_available(
    relative_path: str,
):
    path = _private_reference(relative_path)
    assert json.loads(path.read_text(encoding="utf-8-sig")) is not None


def _normalised_missions(record: dict) -> list[dict]:
    if "missions" in record:
        return record["missions"]
    return [
        {
            "name": record.get("mission"),
            "address": record.get("address"),
            "phone": record.get("phone"),
            "fax": record.get("fax"),
            "email": record.get("email"),
            "website": record.get("website"),
        }
    ]


@pytest.mark.parametrize(
    "relative_path",
    [
        "extra_info/Malaysia_Embassy_Complete.json",
        "extra_info/malaysian_missions_complete.json",
        "fyp_backend/Malaysia_Embassy_Complete.json",
    ],
)
def test_embassy_datasets_cover_every_country_and_mission(relative_path: str):
    if relative_path.startswith("extra_info/"):
        _private_reference(relative_path)
    records = _load_json(relative_path)
    assert len(records) == 82
    assert len({record["country"] for record in records}) == 82

    missions = [
        mission for record in records for mission in _normalised_missions(record)
    ]
    assert len(missions) == 109
    for mission in missions:
        assert {"name", "address"}.issubset(mission)
        assert isinstance(mission["name"], str) and mission["name"].strip()
        assert any(mission.get(field) for field in ("phone", "email", "website"))


def test_backend_embassy_copy_matches_the_canonical_reference_byte_for_byte():
    canonical = _private_reference("extra_info/Malaysia_Embassy_Complete.json")
    assert (
        BACKEND / "Malaysia_Embassy_Complete.json"
    ).read_bytes() == canonical.read_bytes()


class _EmbassyFragmentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.country_headers = 0
        self.mission_headers = 0
        self._inside_paragraph = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "div" and attr_map.get("class") == "link":
            self.country_headers += 1
        if tag == "p":
            self._inside_paragraph = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self._inside_paragraph = False

    def handle_data(self, data: str) -> None:
        mission_labels = (
            "Embassy of Malaysia",
            "High Commission of Malaysia",
            "Consulate General of Malaysia",
            "Permanent Mission of Malaysia",
        )
        if self._inside_paragraph and any(label in data for label in mission_labels):
            self.mission_headers += 1


def test_embassy_html_snapshot_covers_all_country_records():
    html_path = _private_reference("extra_info/all_embassy.html")
    canonical = _private_reference("extra_info/Malaysia_Embassy_Complete.json")
    html = html_path.read_text(encoding="utf-8-sig")
    countries = [
        record["country"]
        for record in json.loads(canonical.read_text(encoding="utf-8-sig"))
    ]
    assert all(country.casefold() in html.casefold() for country in countries)

    parser = _EmbassyFragmentParser()
    parser.feed(html)
    assert parser.country_headers == 82
    assert parser.mission_headers >= 109


@pytest.mark.parametrize("filename", ["fyp.pdf", "fyp_info.pdf"])
def test_reference_pdfs_have_valid_pdf_envelopes(filename: str):
    content = _private_reference(f"extra_info/{filename}").read_bytes()
    assert content.startswith(b"%PDF-1.7")
    assert content.rstrip().endswith(b"%%EOF")
    assert len(content) > 1_000_000


@pytest.mark.parametrize(
    "relative_path",
    [
        "fyp_frontend/index.html",
        "fyp_frontend/src/main.tsx",
        "fyp_frontend/src/app/App.tsx",
        "fyp_backend/main.py",
        "fyp_backend/requirements.txt",
        "tests/run_all.ps1",
    ],
)
def test_primary_entrypoints_are_nonempty_utf8(relative_path: str):
    text = (WORKSPACE / relative_path).read_text(encoding="utf-8-sig")
    assert text.strip()
