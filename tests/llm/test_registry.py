import subprocess
import sys
from types import MappingProxyType

import pytest

from assay.llm import (
    LLMError,
    MissingProviderExtraError,
    OllamaClient,
    Provider,
    RetryingClient,
    TokenPrices,
    UnsupportedProviderError,
    create_client,
    registry,
)

BEDROCK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"


@pytest.fixture
def no_aws_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the developer's AWS profile and credentials out of the test.
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "does-not-exist")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "does-not-exist")


@pytest.mark.parametrize("provider", [Provider.OLLAMA, "ollama"])
def test_creates_a_retrying_ollama_client(provider: Provider | str) -> None:
    client = create_client(provider, "phi4-mini")

    assert isinstance(client, RetryingClient)
    assert client.model_id == "phi4-mini"


@pytest.mark.usefixtures("no_aws_config")
def test_creates_a_bedrock_client_with_provider_options() -> None:
    client = create_client(
        "bedrock",
        BEDROCK_MODEL,
        options={"region": "us-east-1", "context_tokens": 200_000},
    )

    assert isinstance(client, RetryingClient)
    assert (client.model_id, client.model_revision) == (BEDROCK_MODEL, BEDROCK_MODEL)
    assert client.context_tokens == 200_000


def test_creates_an_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = create_client(
        Provider.OPENAI, "gpt-4o-2024-08-06", options={"prices": TokenPrices(2.5, 10.0)}
    )

    assert isinstance(client, RetryingClient)
    assert client.model_id == "gpt-4o-2024-08-06"


def test_a_missing_openai_key_fails_when_the_client_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        create_client("openai", "gpt-4o-2024-08-06")


def test_options_a_client_does_not_take_are_rejected() -> None:
    with pytest.raises(TypeError, match="region"):
        create_client("ollama", "phi4-mini", options={"region": "us-east-1"})


@pytest.mark.parametrize(
    ("provider", "sdk", "module"),
    [("bedrock", "boto3", "assay.llm.bedrock"), ("openai", "openai", "assay.llm.openai")],
)
def test_a_missing_extra_names_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch, provider: str, sdk: str, module: str
) -> None:
    monkeypatch.setitem(sys.modules, sdk, None)
    monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(MissingProviderExtraError, match=f"uv sync --extra {provider}"):
        create_client(provider, "some-model")


def test_other_missing_modules_are_not_blamed_on_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "assay.llm.bedrock", None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        create_client("bedrock", "some-model")

    assert not isinstance(excinfo.value, MissingProviderExtraError)


def test_the_core_install_imports_without_the_provider_extras() -> None:
    blocked = "import sys; sys.modules.update(boto3=None, botocore=None, openai=None); "
    result = subprocess.run(
        [sys.executable, "-c", blocked + "import assay.llm"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_known_providers_without_a_client_name_the_supported_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_PROVIDERS", MappingProxyType({Provider.OLLAMA: OllamaClient}))

    with pytest.raises(UnsupportedProviderError, match=r"'bedrock' yet \(supported: ollama\)"):
        create_client("bedrock", "some-model")


def test_unknown_providers_are_rejected() -> None:
    with pytest.raises(ValueError, match="'cohere' is not a valid Provider"):
        create_client("cohere", "some-model")
