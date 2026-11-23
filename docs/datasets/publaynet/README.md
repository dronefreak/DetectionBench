# PubLayNet: Dataset Statistics

## About

PubLayNet is a large-scale document layout analysis benchmark: 335,703 training / 11,245 validation page images automatically annotated by matching the PDF and XML representations of over 1 million PubMed Central Open Access articles, across five layout element types. It's used to benchmark document layout detection at a scale few manually-annotated datasets can match.

Computed from the canonical COCO layout produced by `detectionbench-prepare-coco --dataset publaynet`. See the adapter and Hydra config for how these splits are built.

## Split Summary

No local copy has been prepared yet -- the full release is 100GB+ (IBM's own hosting is deprecated; use a third-party Hugging Face/Kaggle re-upload), so this project has deliberately not downloaded/converted it. The published splits are:

| Split | Images |
| :--- | ---: |
| train | 335,703 |
| val | 11,245 |
| test | 11,405 (unlabeled -- ICDAR 2021 competition set, not converted) |

Once staged and run through `detectionbench-prepare-coco --dataset publaynet`, regenerate this report with real per-class/per-box numbers via the command at the bottom of this file.

## Class Distribution

Not available yet (no local data). Classes: `text`, `title`, `list`, `table`, `figure`.

## References

**Citation:**

```bibtex
@inproceedings{zhong2019publaynet,
  title={PubLayNet: largest dataset ever for document layout analysis},
  author={Zhong, Xu and Tang, Jianbin and Yepes, Antonio Jimeno},
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages={1015--1022},
  year={2019},
  organization={IEEE}
}
```

- Project website: <https://github.com/ibm-aur-nlp/PubLayNet>

---

Regenerate this report with:

```bash
python -m detectionbench.scripts.dataset_stats --coco-dir <publaynet_coco> --dataset publaynet --output-dir docs/datasets/publaynet
```
