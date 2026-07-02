from biomedqa.corpus import Passage, Question
from biomedqa.evaluate import retrieval_metrics
from biomedqa.retrieve import BM25Retriever


class FixedRetriever:
    """Returns a predetermined ranking, so metric maths can be checked exactly."""
    name = "fixed"

    def __init__(self, ranking):
        self.ranking = ranking

    def index(self, passages):
        return self

    def search(self, query, k=10):
        return [(pid, 1.0) for pid in self.ranking[:k]]


def test_metrics_exact():
    # one gold passage "A"; retriever ranks it 2nd
    questions = [Question("1", "q", "yes", "", None)]
    gold = {"1": {"A"}}
    r = FixedRetriever(["B", "A", "C", "D"])
    m = retrieval_metrics(r, questions, gold, ks=(1, 2, 3))
    assert m["recall_at_k"][1] == 0.0        # A not in top-1
    assert m["recall_at_k"][2] == 1.0        # A in top-2
    assert m["precision_at_k"][2] == 0.5     # 1 of 2 is gold
    assert abs(m["mrr"] - 0.5) < 1e-9        # first gold at rank 2


def test_bm25_finds_relevant_passage():
    passages = [
        Passage("p0", "d", "S", "aspirin reduces the risk of heart attack"),
        Passage("p1", "d", "S", "the migratory patterns of arctic terns"),
        Passage("p2", "d", "S", "photosynthesis in tropical plants"),
    ]
    r = BM25Retriever().index(passages)
    top = r.search("does aspirin lower heart attack risk", k=1)
    assert top[0][0] == "p0"
