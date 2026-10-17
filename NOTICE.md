# Sources and attribution

- OpenLORIS-Scene: Shi et al., “Are We Ready for Service Robots? The OpenLORIS-Scene Datasets for Lifelong SLAM”, ICRA 2020, pp. 3139–3145. https://lifelong-robotic-vision.github.io/dataset/scene.html
- NCLT: Carlevaris-Bianco, Ushani and Eustice, “University of Michigan North Campus long-term vision and lidar dataset”, IJRR 35(9), 2016. https://robots.engin.umich.edu/nclt/
- Cartographer: Hess et al., “Real-Time Loop Closure in 2D LIDAR SLAM”, ICRA 2016. https://github.com/cartographer-project/cartographer
- 2DLIW-SLAM: Zhang et al., Measurement Science and Technology 35(7), 075205, 2024. https://doi.org/10.1088/1361-6501/ad3ea3 ; https://github.com/LittleDang/2DLIW-SLAM

The MIT license covers the original project implementation and documentation. It does not relicense dataset-derived numeric records, third-party dependencies, or upstream software. Raw datasets must be obtained from their authors and used under their applicable terms. External SLAM source is fetched at the commits in `third_party/versions.json`, with upstream notices retained. The 2DLIW-SLAM Apache 2.0 license is preserved in `third_party/LICENSE-2DLIW-SLAM`.

`artifacts/source_provenance.json` records source hashes and the limited portability edits. `reproduction/C20` retains original frozen inference artifacts; its old hosting wording is historical metadata. Cached Python bytecode from the private archive is excluded from the public distribution and corresponding checksum entries are removed. `reproduction/transfer/transfer_source_map.json` maps historical source identifiers to files actually supplied here.

Original Torch checkpoints contain historical dataset-path metadata. Inference uses their frozen network/normalization/gate values, not those absolute paths. Use the published manifests and documented commands for current file locations.
