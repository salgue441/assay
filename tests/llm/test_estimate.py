from assay.llm import ChatRequest, estimate_tokens


def test_estimate_rounds_up_at_three_characters_per_token() -> None:
    assert (estimate_tokens(""), estimate_tokens("abc"), estimate_tokens("abcd")) == (0, 1, 2)


def test_request_estimate_counts_each_message_and_the_reply_budget() -> None:
    user_only = ChatRequest(user="x" * 30)
    with_system = ChatRequest(user="x" * 30, system="y" * 30, max_output_tokens=100)

    assert user_only.estimated_tokens() == 10 + 16
    assert with_system.estimated_tokens() == 2 * (10 + 16) + 100
