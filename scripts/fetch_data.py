"""Download the PubMedQA labelled set (ori_pqal.json) into data/.

The full file is not committed; this pulls it from the official PubMedQA repository.
Source: Jin et al. (2019), https://github.com/pubmedqa/pubmedqa (MIT licensed).
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
DEST = os.path.join(os.path.dirname(__file__), "..", "data", "ori_pqal.json")


def main() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {os.path.normpath(DEST)}")


if __name__ == "__main__":
    main()
