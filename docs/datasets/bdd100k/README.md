# BDD100K: Dataset Statistics

## About

BDD100K is a large-scale, diverse driving-video dataset; this adapter covers its 100K-image object-detection task: 79,863 labelled 1280x720 dashcam images (69,863 train + 10,000 val, the official held-out test split has no released labels) annotated for 10 classes spanning vulnerable road users (person, rider), vehicles (car, truck, bus, train, motor, bike), and traffic control (traffic light, traffic sign), captured across diverse weather, time-of-day, and scene conditions in the US. It's used to benchmark multi-class driving-scene detection under real-world distribution shift.

Computed from the canonical COCO layout produced by `detectionbench-prepare-coco --dataset bdd100k`. See the adapter and Hydra config for how these splits are built.

## Split Summary

| Split | Images | Instances | Instances / Image |
| :--- | ---: | ---: | ---: |
| train | 59,384 | 1,091,589 | 18.38 |
| valid | 10,479 | 195,282 | 18.64 |
| test | 10,000 | 185,526 | 18.55 |
| **Total** | **79,863** | **1,472,397** | **18.44** |

## Class Distribution

![Class distribution](class_distribution.png)

| Class | Instances | Share |
| :--- | ---: | ---: |
| car | 815,717 | 55.4% |
| traffic sign | 274,594 | 18.6% |
| traffic light | 213,002 | 14.5% |
| person | 104,611 | 7.1% |
| truck | 34,216 | 2.3% |
| bus | 13,269 | 0.9% |
| bike | 8,217 | 0.6% |
| rider | 5,166 | 0.4% |
| motor | 3,454 | 0.2% |
| train | 151 | 0.0% |

## Bounding Box Geometry

- Median box area: **0.09%** of image area (mean 0.73%)
- Instances per image: mean **18.44**, median **17**, max **91**

## References

**Citation:**

```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2633--2642},
  year={2020}
}
```

- Project website: <https://www.bdd100k.com/>

---

Regenerate this report with:

```bash
python -m detectionbench.scripts.dataset_stats --coco-dir <bdd100k_coco> --dataset bdd100k --output-dir docs/datasets/bdd100k
```
