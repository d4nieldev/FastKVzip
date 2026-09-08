import pytest

from data.benchmarks import get_data_list, parse_ruler_name


def test_all_selects_each_published_benchmark_once_without_model_substitution():
    names = get_data_list("all")

    assert len(names) == len(set(names)) == 107
    assert sum(name.startswith("scbench_") for name in names) == 27
    assert sum(name.startswith("ruler_") for name in names) == 78
    assert {"squad", "gsm", "scbench_kv", "scbench_kv_mid"} <= set(names)
    assert "agentic" not in names
    with pytest.raises(TypeError):
        get_data_list("all", "qwen3")


def test_ruler_selectors_expand_lengths_and_validate_concrete_names():
    assert len(get_data_list("ruler")) == 78
    assert len(get_data_list("ruler_4k")) == 13
    assert get_data_list("ruler_qa_1_128k") == ["ruler_qa_1_128k"]
    assert parse_ruler_name("ruler_niah_single_1_4k") == ("niah_single_1", "4k")
    for name in ("ruler_qa_3_4k", "ruler_qa_1_3k", "ruler_unknown"):
        with pytest.raises(ValueError, match="RULER"):
            get_data_list(name)


def test_existing_convenience_groups_and_explicit_names_remain_available():
    assert get_data_list("short") == ["squad", "gsm"]
    assert len(get_data_list("mid")) == 5
    assert len(get_data_list("long")) == 4
    assert len(get_data_list("multi")) == 2
    assert get_data_list("scbench_kv") == ["scbench_kv"]
