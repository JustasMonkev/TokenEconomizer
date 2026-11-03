from tokeneconomizer.toolprune.tools import judge_tools_for_task


def test_equivalent_catalogs_produce_identical_cacheable_prompts():
    prompts = []

    def runner(prompt, model):
        prompts.append(prompt)
        return '{}', 7

    catalog = {"WebSearch": "search the web", "WebFetch": "fetch a URL"}
    reversed_catalog = dict(reversed(list(catalog.items())))
    assert judge_tools_for_task("Fix parsing.", catalog, runner) == ({}, 7)
    assert judge_tools_for_task("Fix parsing.", reversed_catalog, runner) == ({}, 7)
    assert prompts[0] == prompts[1]
