import pytest

from assay.llm import Provider, RetryingClient, UnsupportedProviderError, create_client


@pytest.mark.parametrize("provider", [Provider.OLLAMA, "ollama"])
def test_creates_a_retrying_ollama_client(provider: Provider | str) -> None:
    client = create_client(provider, "phi4-mini")

    assert isinstance(client, RetryingClient)
    assert client.model_id == "phi4-mini"


def test_known_providers_without_a_client_name_the_supported_ones() -> None:
    with pytest.raises(UnsupportedProviderError, match=r"'bedrock' yet \(supported: ollama\)"):
        create_client("bedrock", "some-model")


def test_unknown_providers_are_rejected() -> None:
    with pytest.raises(ValueError, match="'cohere' is not a valid Provider"):
        create_client("cohere", "some-model")
