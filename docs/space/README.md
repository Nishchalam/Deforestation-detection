---
title: Deforestation explorer
emoji: 🌳
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Deforestation explorer

Pick an area, get forest masks and a forest-loss map from Sentinel-2 composites, computed with a U-Net exported to ONNX and compared with the Hansen layer.

**Caveat:** the model was trained only on Ariquemes, Rondônia (Brazil), 2018 vs 2022. Accuracy elsewhere is unmeasured.

Source, method and results: https://github.com/Nishchalam/Deforestation-detection
