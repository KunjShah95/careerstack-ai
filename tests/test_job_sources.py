"""Tests for the job source adapters and the multi-source fan-out.

No network: each adapter's HTTP layer is stubbed with a payload in the
real provider's shape (captured from live responses while building these
adapters), and the point is the normalisation, the fan-out resilience and
the dedupe priority -- not the providers themselves.
"""

import asyncio

import httpx
import pytest

from app.services import discovery
from app.services.jobs import adzuna_source, greenhouse, jsearch
from app.services.jobs.base import SOURCE_PRIORITY, raw_posting, source_rank

# --------------------------------------------------------------------------
# Common shape
# --------------------------------------------------------------------------

COMMON_FIELDS = {
    "id", "source", "source_label", "title", "company", "location", "description",
    "category", "employment_type", "is_remote", "url", "created",
    "salary_min", "salary_max", "salary_currency",
}


def test_raw_posting_has_the_common_field_set():
    posting = raw_posting(
        source="adzuna",
        external_id="1",
        title="Backend Developer",
        company="Acme",
        location="Pune",
        description="text",
        url="https://example.test",
        created="2026-09-01T00:00:00Z",
    )
    assert set(posting.keys()) == COMMON_FIELDS


def test_source_priority_order_is_jsearch_then_greenhouse_then_adzuna():
    assert SOURCE_PRIORITY == ["jsearch", "greenhouse", "adzuna"]
    assert source_rank("jsearch") < source_rank("greenhouse") < source_rank("adzuna")


def test_unknown_source_sorts_last_rather_than_raising():
    assert source_rank("some-future-source") > source_rank("adzuna")
    assert source_rank(None) > source_rank("adzuna")


# --------------------------------------------------------------------------
# Adzuna adapter
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_adzuna_adapter_normalises_to_the_common_schema(monkeypatch):
    async def fake_search(query, location, limit, country="in", page=1):
        if page > 1:
            return []
        return [
            {
                "id": 123,
                "title": "Backend Developer",
                "company": "Acme Pvt Ltd",
                "location": "Ahmedabad, Gujarat",
                "description": "Python and Django required.",
                "category": "IT Jobs",
                "url": "https://adzuna.test/job/123",
                "created": "2026-09-20T10:00:00Z",
                "salary_min": 600000.0,
                "salary_max": 900000.0,
            }
        ]

    monkeypatch.setattr(adzuna_source, "adzuna_search", fake_search)
    postings = await adzuna_source.fetch("backend developer", "Ahmedabad")

    assert len(postings) == 1
    posting = postings[0]
    assert set(posting.keys()) == COMMON_FIELDS
    assert posting["source"] == "adzuna"
    assert posting["id"] == "123"
    assert posting["title"] == "Backend Developer"
    assert posting["salary_currency"] == "INR"


@pytest.mark.anyio
async def test_adzuna_adapter_reports_no_currency_when_there_is_no_salary(monkeypatch):
    async def fake_search(query, location, limit, country="in", page=1):
        if page > 1:
            return []
        return [{"id": 1, "title": "Dev", "company": "Acme", "location": "Pune",
                 "description": "x", "url": "u", "created": "c",
                 "salary_min": None, "salary_max": None}]

    monkeypatch.setattr(adzuna_source, "adzuna_search", fake_search)
    postings = await adzuna_source.fetch("dev", "Pune")
    assert postings[0]["salary_currency"] is None


# --------------------------------------------------------------------------
# JSearch adapter
# --------------------------------------------------------------------------


# One entry from a real JSearch `data` array, trimmed to the fields read.
JSEARCH_PAYLOAD = {
    "data": [
        {
            "job_id": "abc123",
            "job_title": "Senior Backend Engineer",
            "employer_name": "Globex",
            "job_city": "Bengaluru",
            "job_state": "Karnataka",
            "job_country": "IN",
            "job_description": "Full description text, not a truncated snippet. " * 20,
            "job_apply_link": "https://jsearch.test/job/abc123",
            "job_posted_at_datetime_utc": "2026-09-18T08:00:00.000Z",
            "job_min_salary": 1800000,
            "job_max_salary": 2600000,
            "job_salary_currency": "INR",
            "job_employment_type": "FULLTIME",
            "job_is_remote": False,
        }
    ]
}


@pytest.mark.anyio
async def test_jsearch_adapter_normalises_to_the_common_schema(monkeypatch):
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")

    async def fake_get(params, headers):
        assert headers["X-RapidAPI-Host"] == "jsearch.p.rapidapi.com"
        assert headers["X-RapidAPI-Key"] == "test-key"
        assert params["query"] == "backend developer in Bengaluru"
        assert params["date_posted"] == "month"
        return httpx.Response(200, json=JSEARCH_PAYLOAD)

    monkeypatch.setattr(jsearch, "_get", fake_get)
    postings = await jsearch.fetch("backend developer", "Bengaluru")

    assert len(postings) == 1
    posting = postings[0]
    assert set(posting.keys()) == COMMON_FIELDS
    assert posting["source"] == "jsearch"
    assert posting["company"] == "Globex"
    assert posting["location"] == "Bengaluru, Karnataka"
    assert posting["salary_currency"] == "INR"
    assert posting["employment_type"] == "FULLTIME"
    assert posting["is_remote"] is False
    # Terms attribution travels with the posting.
    assert posting["source_label"] == "via JSearch"
    # The reason this source outranks the others.
    assert len(posting["description"]) > 500


@pytest.mark.anyio
async def test_jsearch_without_an_api_key_returns_empty_and_does_not_raise(monkeypatch):
    """The configured state in this repo: RAPIDAPI_KEY is empty, so this
    path runs on every discovery call and must be silent and harmless.
    """
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "")

    async def exploding_get(params, headers):
        raise AssertionError("must not make a request without a key")

    monkeypatch.setattr(jsearch, "_get", exploding_get)

    assert await jsearch.fetch("backend developer", "India") == []
    assert jsearch.is_configured() is False


@pytest.mark.anyio
async def test_jsearch_rate_limit_returns_empty_rather_than_raising(monkeypatch):
    """429 is expected on a ~200 call/month tier and must not fail a run."""
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")

    async def rate_limited(params, headers):
        raise httpx.HTTPStatusError(
            "429", request=httpx.Request("GET", "https://x"), response=httpx.Response(429)
        )

    monkeypatch.setattr(jsearch, "_get", rate_limited)
    assert await jsearch.fetch("backend developer", "India") == []


# --------------------------------------------------------------------------
# Greenhouse adapter
# --------------------------------------------------------------------------


def test_greenhouse_unescapes_and_strips_html_from_descriptions():
    """Boards return HTML-escaped markup. Unescaping must happen before
    tag-stripping, or `&lt;div&gt;` survives as literal text.
    """
    raw = "&lt;div&gt;&lt;strong&gt;About:&lt;/strong&gt;&lt;/div&gt;\n&lt;p&gt;We use Python.&lt;/p&gt;"
    cleaned = greenhouse.clean_description(raw)

    assert "<" not in cleaned and "&lt;" not in cleaned
    assert "About:" in cleaned
    assert "We use Python." in cleaned


def test_greenhouse_clean_description_handles_empty_input():
    assert greenhouse.clean_description(None) == ""
    assert greenhouse.clean_description("") == ""


@pytest.mark.parametrize(
    "job_location, wanted",
    [
        ("Bengaluru, India", "bangalore"),
        ("Bengaluru, India", "BANGALORE"),
        ("Bengaluru, India", "Bengaluru"),
        ("BENGALURU, INDIA", "bangalore"),
        ("Mumbai, India", "mumbai"),
        ("Mumbai, India", "Bombay"),
        ("Gurugram, India", "gurgaon"),
        ("Pune, India", "PUNE"),
    ],
)
def test_greenhouse_city_filter_is_case_insensitive(job_location, wanted):
    assert greenhouse.matches_location(job_location, wanted) is True


def test_greenhouse_city_filter_rejects_a_different_city():
    assert greenhouse.matches_location("Bengaluru, India", "Ahmedabad") is False
    assert greenhouse.matches_location("New York, USA", "Mumbai") is False


def test_greenhouse_country_wide_search_matches_any_india_location():
    assert greenhouse.matches_location("Bengaluru, India", "India") is True
    assert greenhouse.matches_location("Mumbai, India", "india") is True
    # A non-India posting on the same board is excluded.
    assert greenhouse.matches_location("San Francisco, USA", "India") is False


GREENHOUSE_PAYLOAD = {
    "jobs": [
        {
            "id": 4970739101,
            "title": "Backend Engineer",
            "location": {"name": "Bengaluru, India"},
            "content": "&lt;div&gt;We use Python and Django.&lt;/div&gt;",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/4970739101",
            "first_published": "2026-09-14T03:49:28-04:00",
            "updated_at": "2026-09-14T03:52:53-04:00",
            "company_name": "Acme",
            "departments": [{"name": "Engineering"}],
        },
        {
            "id": 2,
            "title": "Sales Lead",
            "location": {"name": "San Francisco, USA"},
            "content": "&lt;p&gt;Not in India.&lt;/p&gt;",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/2",
            "first_published": "2026-09-10T00:00:00-04:00",
            "company_name": "Acme",
            "departments": [],
        },
    ]
}


@pytest.mark.anyio
async def test_greenhouse_adapter_normalises_and_filters_by_location(monkeypatch):
    monkeypatch.setattr(greenhouse, "load_company_tokens", lambda: ["acme"])

    async def fake_get_board(client, token):
        return GREENHOUSE_PAYLOAD["jobs"]

    monkeypatch.setattr(greenhouse, "_get_board", fake_get_board)
    postings = await greenhouse.fetch("backend developer", "Bangalore")

    assert len(postings) == 1  # the SF posting is filtered out
    posting = postings[0]
    assert set(posting.keys()) == COMMON_FIELDS
    assert posting["source"] == "greenhouse"
    assert posting["title"] == "Backend Engineer"
    assert "We use Python and Django." in posting["description"]
    # Terms compliance: links back to the board.
    assert posting["url"].startswith("https://job-boards.greenhouse.io/")


@pytest.mark.anyio
async def test_greenhouse_one_failing_board_does_not_kill_the_fan_out(monkeypatch):
    monkeypatch.setattr(greenhouse, "load_company_tokens", lambda: ["dead", "alive"])

    async def fake_get_board(client, token):
        if token == "dead":
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", "https://x"), response=httpx.Response(404)
            )
        return GREENHOUSE_PAYLOAD["jobs"]

    monkeypatch.setattr(greenhouse, "_get_board", fake_get_board)
    postings = await greenhouse.fetch("backend developer", "India")

    # The live board still contributed despite the dead one.
    assert len(postings) == 1
    assert postings[0]["company"] == "Acme"


@pytest.mark.anyio
async def test_greenhouse_with_no_configured_tokens_returns_empty(monkeypatch):
    monkeypatch.setattr(greenhouse, "load_company_tokens", lambda: [])
    assert await greenhouse.fetch("backend developer", "India") == []


def test_every_shipped_greenhouse_token_is_a_nonempty_string():
    tokens = greenhouse.load_company_tokens()
    assert tokens, "the curated company list should not be empty"
    assert all(isinstance(token, str) and token for token in tokens)
    assert len(tokens) == len(set(tokens)), "duplicate board tokens"


# --------------------------------------------------------------------------
# Fan-out and dedupe priority
# --------------------------------------------------------------------------


def _stub_sources(monkeypatch, adzuna=None, jsearch_result=None, greenhouse_result=None,
                  adzuna_raises=False, greenhouse_raises=False):
    async def fake_adzuna(role, location, limit=None):
        if adzuna_raises:
            raise RuntimeError("adzuna exploded")
        return adzuna or []

    async def fake_jsearch(role, location, limit=50):
        return jsearch_result or []

    async def fake_greenhouse(role, location, limit=200):
        if greenhouse_raises:
            raise RuntimeError("greenhouse exploded")
        return greenhouse_result or []

    monkeypatch.setattr(discovery.adzuna_source, "fetch", fake_adzuna)
    monkeypatch.setattr(discovery.jsearch, "fetch", fake_jsearch)
    monkeypatch.setattr(discovery.greenhouse, "fetch", fake_greenhouse)
    # Bypass the on-disk source cache so these exercise the fan-out itself.
    monkeypatch.setattr(discovery, "_cached_source_postings", lambda *a, **k: None)
    monkeypatch.setattr(discovery, "_store_source_postings", lambda *a, **k: None)


def _src_posting(source, title="Backend Developer", company="Acme", description="Python.", **kw):
    return raw_posting(
        source=source,
        external_id=kw.get("external_id", f"{source}-1"),
        title=title,
        company=company,
        location=kw.get("location", "Bengaluru, India"),
        description=description,
        url=f"https://{source}.test/job",
        created=kw.get("created", "2026-09-20T00:00:00Z"),
    )


@pytest.mark.anyio
async def test_fan_out_gathers_all_three_sources_and_counts_each(monkeypatch):
    _stub_sources(
        monkeypatch,
        adzuna=[_src_posting("adzuna", company="A1"), _src_posting("adzuna", company="A2")],
        jsearch_result=[_src_posting("jsearch", company="J1")],
        greenhouse_result=[_src_posting("greenhouse", company="G1")],
    )

    postings, counts = await discovery._gather_sources(
        "backend developer", ["backend developer"], "Bengaluru"
    )

    assert counts == {"adzuna": 2, "jsearch": 1, "greenhouse": 1}
    assert len(postings) == 4


@pytest.mark.anyio
async def test_fan_out_completes_when_one_source_errors(monkeypatch):
    """The central resilience guarantee: a source raising outright must
    cost only its own results.
    """
    _stub_sources(
        monkeypatch,
        adzuna_raises=True,
        jsearch_result=[_src_posting("jsearch", company="J1")],
        greenhouse_result=[_src_posting("greenhouse", company="G1")],
    )

    postings, counts = await discovery._gather_sources(
        "backend developer", ["backend developer"], "India"
    )

    assert counts["adzuna"] == 0
    assert counts["jsearch"] == 1
    assert counts["greenhouse"] == 1
    assert len(postings) == 2


@pytest.mark.anyio
async def test_fan_out_completes_when_every_source_errors(monkeypatch):
    _stub_sources(monkeypatch, adzuna_raises=True, greenhouse_raises=True)

    postings, counts = await discovery._gather_sources("role", ["role"], "India")

    assert postings == []
    assert counts == {"adzuna": 0, "jsearch": 0, "greenhouse": 0}


@pytest.mark.anyio
async def test_missing_rapidapi_key_does_not_break_the_run(monkeypatch):
    """End-to-end version of the JSearch no-key path, through the fan-out:
    the other two sources still produce a full result set.
    """
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "")
    monkeypatch.setattr(discovery, "_cached_source_postings", lambda *a, **k: None)
    monkeypatch.setattr(discovery, "_store_source_postings", lambda *a, **k: None)

    async def fake_adzuna(role, location, limit=None):
        return [_src_posting("adzuna", company="A1")]

    async def fake_greenhouse(role, location, limit=200):
        return [_src_posting("greenhouse", company="G1")]

    monkeypatch.setattr(discovery.adzuna_source, "fetch", fake_adzuna)
    monkeypatch.setattr(discovery.greenhouse, "fetch", fake_greenhouse)

    postings, counts = await discovery._gather_sources("role", ["role"], "India")

    assert counts["jsearch"] == 0
    assert counts["adzuna"] == 1 and counts["greenhouse"] == 1
    assert len(postings) == 2


def test_jsearch_wins_over_adzuna_for_an_identical_fingerprint():
    """Same job, same fingerprint, two sources. JSearch must survive dedupe
    because its description is the full posting rather than a snippet.
    """
    adzuna_job = discovery.normalise_posting(
        _src_posting("adzuna", description="Truncated snippet...")
    )
    jsearch_job = discovery.normalise_posting(
        _src_posting("jsearch", description="The full posting text, at length. " * 10)
    )
    assert adzuna_job.fingerprint == jsearch_job.fingerprint

    # Adzuna arrives first, so first-seen-wins would keep the wrong one.
    unique, duplicates = discovery.dedupe([adzuna_job, jsearch_job])

    assert len(unique) == 1
    assert duplicates == 1
    assert unique[0].source == "jsearch"


def test_greenhouse_wins_over_adzuna_but_loses_to_jsearch():
    adzuna_job = discovery.normalise_posting(_src_posting("adzuna"))
    greenhouse_job = discovery.normalise_posting(_src_posting("greenhouse"))
    jsearch_job = discovery.normalise_posting(_src_posting("jsearch"))

    unique, _ = discovery.dedupe([adzuna_job, greenhouse_job])
    assert unique[0].source == "greenhouse"

    unique, _ = discovery.dedupe([greenhouse_job, jsearch_job])
    assert unique[0].source == "jsearch"

    unique, _ = discovery.dedupe([jsearch_job, greenhouse_job, adzuna_job])
    assert len(unique) == 1
    assert unique[0].source == "jsearch"


def test_source_priority_beats_description_length():
    """Priority is the primary key: a short JSearch description still wins
    over a long Adzuna one, because the source is the better authority.
    """
    long_adzuna = discovery.normalise_posting(
        _src_posting("adzuna", description="x" * 5000)
    )
    short_jsearch = discovery.normalise_posting(
        _src_posting("jsearch", description="short")
    )

    unique, _ = discovery.dedupe([long_adzuna, short_jsearch])
    assert unique[0].source == "jsearch"


def test_normalise_posting_keeps_the_adapter_source():
    for source in ("adzuna", "jsearch", "greenhouse"):
        job = discovery.normalise_posting(_src_posting(source))
        assert job.source == source


def test_normalise_posting_defaults_to_adzuna_for_a_sourceless_dict():
    """Hand-built dicts in the older tests carry no `source` key; they must
    still validate rather than failing the model.
    """
    job = discovery.normalise_posting({"title": "Dev", "company": "Acme", "location": "Pune"})
    assert job.source == "adzuna"


# --------------------------------------------------------------------------
# JSearch request parameters and quota budget
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_jsearch_sends_every_required_parameter(monkeypatch):
    """The exact param set matters: country scopes to India, num_pages=2
    buys ~20 results for one quota call, and work_from_home must NOT be
    sent -- remote is a client-side filter on the discovery page, so
    narrowing it here would hide postings the user can still choose.
    """
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")
    monkeypatch.setattr(jsearch.quota, "has_budget", lambda *a, **k: True)
    monkeypatch.setattr(jsearch.quota, "record_call", lambda *a, **k: 1)

    captured = {}

    async def fake_get(params, headers):
        captured.update(params)
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(jsearch, "_get", fake_get)
    await jsearch.fetch("web developer", "Ahmedabad")

    assert captured["query"] == "web developer in Ahmedabad"
    assert captured["country"] == "in"
    assert captured["page"] == "1"
    assert captured["num_pages"] == "2"
    assert captured["date_posted"] == "month"
    assert captured["language"] == "en"
    assert "work_from_home" not in captured


@pytest.mark.anyio
async def test_jsearch_counts_every_call_against_the_monthly_budget(monkeypatch):
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")
    recorded = []
    monkeypatch.setattr(jsearch.quota, "has_budget", lambda *a, **k: True)
    monkeypatch.setattr(jsearch.quota, "record_call", lambda source, **k: recorded.append(source) or 1)

    async def fake_get(params, headers):
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(jsearch, "_get", fake_get)
    await jsearch.fetch("dev", "India")

    assert recorded == ["jsearch"]


@pytest.mark.anyio
async def test_jsearch_skips_the_call_once_the_budget_is_spent(monkeypatch):
    """At MONTHLY_CALL_BUDGET the source stops itself, preserving the
    reserve for a demo -- and makes no HTTP request at all.
    """
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")
    monkeypatch.setattr(jsearch.quota, "has_budget", lambda *a, **k: False)

    async def exploding_get(params, headers):
        raise AssertionError("must not call the API with no budget left")

    monkeypatch.setattr(jsearch, "_get", exploding_get)
    assert await jsearch.fetch("dev", "India") == []


@pytest.mark.anyio
async def test_jsearch_counts_a_failed_call_too(monkeypatch):
    """A request that fails still consumed quota upstream, so it must be
    counted -- counting only successes drifts optimistic and overruns.
    """
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")
    recorded = []
    monkeypatch.setattr(jsearch.quota, "has_budget", lambda *a, **k: True)
    monkeypatch.setattr(jsearch.quota, "record_call", lambda source, **k: recorded.append(source) or 1)

    async def failing_get(params, headers):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(jsearch, "_get", failing_get)
    assert await jsearch.fetch("dev", "India") == []
    assert recorded == ["jsearch"]


def test_jsearch_budget_reserves_calls_below_the_real_tier_limit():
    assert jsearch.MONTHLY_CALL_BUDGET < jsearch.MONTHLY_CALL_LIMIT
    assert jsearch.MONTHLY_CALL_LIMIT - jsearch.MONTHLY_CALL_BUDGET == jsearch.MONTHLY_CALL_RESERVE


# --------------------------------------------------------------------------
# Quota counter
# --------------------------------------------------------------------------


def test_quota_counts_and_resets_per_calendar_month():
    from datetime import datetime, timezone

    from app.services.jobs import quota

    september = datetime(2026, 9, 15, tzinfo=timezone.utc)
    october = datetime(2026, 10, 1, tzinfo=timezone.utc)
    source = "quota-test-source"

    assert quota.usage(source, now=september) == 0
    assert quota.record_call(source, now=september) == 1
    assert quota.record_call(source, now=september) == 2
    assert quota.usage(source, now=september) == 2

    # New month, new key -- the reset needs no scheduled job.
    assert quota.usage(source, now=october) == 0
    assert quota.record_call(source, now=october) == 1
    # September's record is preserved, not overwritten.
    assert quota.usage(source, now=september) == 2


def test_quota_has_budget_respects_the_limit():
    from datetime import datetime, timezone

    from app.services.jobs import quota

    now = datetime(2026, 11, 5, tzinfo=timezone.utc)
    source = "budget-test-source"

    assert quota.has_budget(source, limit=2, now=now) is True
    quota.record_call(source, now=now)
    assert quota.has_budget(source, limit=2, now=now) is True
    quota.record_call(source, now=now)
    assert quota.has_budget(source, limit=2, now=now) is False


# --------------------------------------------------------------------------
# Remote flag: provider-stated beats inferred
# --------------------------------------------------------------------------


def test_a_provider_stated_remote_flag_beats_text_sniffing():
    """JSearch reports job_is_remote -- the employer's own answer. It must
    win over our regex, which only exists for sources that say nothing.
    """
    # Says remote, but the text never mentions it: trust the provider.
    stated_remote = discovery.normalise_posting(
        raw_posting(
            source="jsearch", external_id="1", title="Engineer", company="Acme",
            location="Bengaluru", description="Build things.", url="u",
            created=None, is_remote=True,
        )
    )
    assert stated_remote.is_remote is True

    # Says NOT remote, though the text mentions the word: still trust it.
    stated_onsite = discovery.normalise_posting(
        raw_posting(
            source="jsearch", external_id="2", title="Engineer", company="Acme",
            location="Bengaluru", description="This is not a remote role.", url="u",
            created=None, is_remote=False,
        )
    )
    assert stated_onsite.is_remote is False


def test_remote_is_inferred_when_the_provider_says_nothing():
    inferred = discovery.normalise_posting(
        raw_posting(
            source="adzuna", external_id="3", title="Engineer", company="Acme",
            location="Remote", description="Fully remote position.", url="u",
            created=None, is_remote=None,
        )
    )
    assert inferred.is_remote is True


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def test_every_source_carries_its_own_attribution_label():
    labels = {
        source: discovery.normalise_posting(
            raw_posting(
                source=source, external_id="1", title="Dev", company="Acme",
                location="Pune", description="x", url="u", created=None,
            )
        ).source_label
        for source in ("jsearch", "greenhouse", "adzuna")
    }
    assert labels == {
        "jsearch": "via JSearch",
        "greenhouse": "via Greenhouse",
        "adzuna": "via Adzuna",
    }


@pytest.mark.anyio
async def test_jsearch_does_not_retry_a_permanent_failure(monkeypatch):
    """403 is RapidAPI's "not subscribed to this API" -- an account fact,
    not a blip. Retrying it three times burned three calls against a
    200/month tier for nothing. Verified against the live API before this
    guard existed.
    """
    monkeypatch.setattr(jsearch.settings, "rapidapi_key", "test-key")
    monkeypatch.setattr(jsearch.quota, "has_budget", lambda *a, **k: True)
    monkeypatch.setattr(jsearch.quota, "record_call", lambda *a, **k: 1)

    attempts = []

    # Patch the transport, not the client: that leaves the module's own
    # @retry decorator in place, so the retry policy is what's under test.
    def forbidden_transport(request):
        attempts.append(1)
        return httpx.Response(403, json={"message": "You are not subscribed to this API."})

    real_init = httpx.AsyncClient.__init__

    def init_with_mock_transport(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(forbidden_transport)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init_with_mock_transport)

    assert await jsearch.fetch("dev", "India") == []
    assert len(attempts) == 1, "a permanent 403 must not be retried"


@pytest.mark.anyio
async def test_jsearch_still_retries_a_transient_failure(monkeypatch):
    """A 500 or a timeout is worth another attempt -- only the permanent
    statuses are exempt.
    """
    assert jsearch._is_retryable(
        httpx.HTTPStatusError("500", request=httpx.Request("GET", "https://x"),
                              response=httpx.Response(500))
    ) is True
    assert jsearch._is_retryable(
        httpx.HTTPStatusError("429", request=httpx.Request("GET", "https://x"),
                              response=httpx.Response(429))
    ) is True
    assert jsearch._is_retryable(httpx.ConnectError("boom")) is True
    for status in (400, 401, 403, 404):
        assert jsearch._is_retryable(
            httpx.HTTPStatusError(str(status), request=httpx.Request("GET", "https://x"),
                                  response=httpx.Response(status))
        ) is False
