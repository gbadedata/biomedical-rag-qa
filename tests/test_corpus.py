from biomedqa.corpus import build_corpus, chunk_text, tokenize

SAMPLE = {
    "111": {
        "QUESTION": "Does X cause Y?",
        "CONTEXTS": ["Background on X.", "Results show X raises Y."],
        "LABELS": ["BACKGROUND", "RESULTS"],
        "final_decision": "Yes",
        "LONG_ANSWER": "X raises Y.",
        "YEAR": "2019",
    },
    "222": {
        "QUESTION": "Is Z safe?",
        "CONTEXTS": ["Z was well tolerated."],
        "LABELS": ["RESULTS"],
        "final_decision": "maybe",
        "LONG_ANSWER": "Unclear.",
        "YEAR": None,
    },
}


def test_build_corpus_shapes():
    passages, gold, questions = build_corpus(SAMPLE)
    assert len(passages) == 3
    assert len(questions) == 2
    assert gold["111"] == {"111#0", "111#1"}
    assert gold["222"] == {"222#0"}


def test_decision_is_lowercased():
    _, _, questions = build_corpus(SAMPLE)
    assert {q.decision for q in questions} == {"yes", "maybe"}


def test_chunk_text_windows():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_tokenize():
    assert tokenize("Anti-TNF, dose 40mg!") == ["anti", "tnf", "dose", "40mg"]
