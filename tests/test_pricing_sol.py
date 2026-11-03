import pytest

from tokeneconomizer import pricing
from tokeneconomizer.plugin import accounting


def test_sol_61_prices_all_token_buckets_without_fallback():
    usage = {"model": "gpt-6.1-sol", "input_tokens": 1000,
             "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 400,
             "output_tokens": 300}
    assert pricing.effective_tokens(usage) == pytest.approx(3100)
    effective, basis = accounting.effective(usage)
    assert effective == pytest.approx(3100)
    assert basis == "model"


def test_unknown_sol_version_is_not_silently_priced():
    with pytest.raises(ValueError, match="no weight entry"):
        pricing.weights_for({"model": "gpt-6.1-sol-unpriced"})
