from __future__ import annotations

import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2] / "fyp_backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Make imports deterministic and prevent tests from depending on or exposing
# values in the developer's real .env file. Clients remain lazy/unopened and
# every I/O boundary is mocked in the tests.
_TEST_ENV = {
    "GEMINI_API_KEY": "test-gemini-key",
    "GEMINI_IATA_API": "test-gemini-key",
    "GEMINI_FLIGHTS_HOTELS": "test-gemini-key",
    "GEMINI_EXTRACTOR_API": "test-gemini-key",
    "GEMINI_EMBED_API": "test-gemini-key",
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_API_KEY": "test-supabase-key",
    "SUPABASE_POSTGRES_URI": "postgresql://test:test@127.0.0.1:5432/test",
    "FIREBASE_CREDENTIAL_JSON": '{"type":"service_account","project_id":"test"}',
    "MAPBOX_TOKEN": "test-mapbox-token",
    "SERPAPI_FLIGHT": "test-serpapi-key",
    "SERPAPI_HOTEL": "test-serpapi-key",
    "SERPAPI_PLACE": "test-serpapi-key",
    "FREECURRENCY_API": "test-currency-key",
    "PINECONE_API_KEY": "test-pinecone-key",
}
for _name, _value in _TEST_ENV.items():
    os.environ[_name] = _value


def pytest_configure(config):
    os.chdir(BACKEND_ROOT)
