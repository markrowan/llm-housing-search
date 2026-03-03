from app.prompt_parser import parse_prompt


def test_example_prompt_parsing():
    prompt = (
        "2.5 to 3.5 rooms, in a green and leafy suburb near Kreis 7 "
        "in Zürich and within 20 minutes to the lake"
    )
    filters = parse_prompt(prompt, use_llm=False)

    assert filters.min_rooms == 2.5
    assert filters.max_rooms == 3.5
    assert filters.max_commute_minutes == 20
    assert filters.commute_target == "lake"
    assert "Kreis 7" in filters.location_tags
    assert "Zürich" in filters.location_tags
    assert "green" in filters.keywords
    assert "leafy" in filters.keywords
