# Based on Open-Prompt-Injection/tasks/gigaword.py

import os
import datasets


_CITATION = """
@article{graff2003english,
  title={English gigaword},
  author={Graff, David and Kong, Junbo and Chen, Ke and Maeda, Kazuaki},
  journal={Linguistic Data Consortium, Philadelphia},
  volume={4},
  number={1},
  pages={34},
  year={2003}
}
"""

_DESCRIPTION = "Gigaword summarization dataset adapted from Open-Prompt-Injection."
_URL = "https://drive.google.com/uc?export=download&id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV"


class Gigaword(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.2.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "document": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                }
            ),
            supervised_keys=("document", "summary"),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Defines dataset splits and handles archive extraction.
        """
        local_archive = os.path.join(
            os.getcwd(),
            "data",
            "corpus",
            "gigaword",
            "gigaword_data.tar.gz",
        )

        if os.path.exists(local_archive):
            print(f"Using local Gigaword archive at {local_archive}")
            extract_path = dl_manager.download_and_extract(local_archive)
        else:
            print("Local Gigaword archive not found, attempting remote download")
            extract_path = dl_manager.download_and_extract(_URL)

        pattern = os.path.join(extract_path, "org_data", "%s.%s.txt")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "src_path": pattern % ("train", "src"),
                    "tgt_path": pattern % ("train", "tgt"),
                    "replace_unk": True,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "src_path": pattern % ("dev", "src"),
                    "tgt_path": pattern % ("dev", "tgt"),
                    "replace_unk": True,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "src_path": pattern % ("test", "src"),
                    "tgt_path": pattern % ("test", "tgt"),
                    "replace_unk": False,
                },
            ),
        ]

    def _generate_examples(self, src_path=None, tgt_path=None, replace_unk=False):
        """
        Yields paired document and summary examples.
        """
        with open(src_path, encoding="utf-8") as src_file, open(
            tgt_path, encoding="utf-8"
        ) as tgt_file:
            for idx, (doc_line, summary_line) in enumerate(
                zip(src_file, tgt_file)
            ):
                document = doc_line.strip()
                summary = summary_line.strip()

                if replace_unk:
                    document = document.replace("<unk>", "UNK")
                    summary = summary.replace("<unk>", "UNK")

                yield idx, {
                    "document": document,
                    "summary": summary,
                }
