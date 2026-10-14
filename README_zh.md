# MAIP：多频电阻抗成像重建

论文 **Multifrequency Electrical Impedance Tomography Reconstruction With
Multibranch Attention Image Prior** 的代码和数值数据。
论文发表于 IEEE Internet of Things Journal，13(2)，1862–1874，2026。

[论文链接](https://ieeexplore.ieee.org/abstract/document/11214340) ·
[DOI](https://doi.org/10.1109/JIOT.2025.3624228) · [数据说明](data/README.md)

MAIP 将四分支注意力网络作为图像先验，每次从随机初始化开始，通过敏感矩阵
将预测图像映射到电压域，再以电压误差更新网络。不需要训练图像或预训练权重。

![四频重建](examples/reconstruction.png)

## 直接运行

使用 Python 3.10 或更新版本，在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python run_demo.py
```

默认使用 `case2_sim`，与原始 `inference.py` 中的 `V_P6.txt` 完全一致。
保留原网络和设置：4 个频率通道、64 × 64 网格、950 次 Adam 迭代、
学习率 `1.2e-4`、随机种子 `2`、电压 L1 损失。有 CUDA 就使用 GPU，否则使用 CPU。
完整重建建议使用 GPU。

结果写入 `results/`：重建 PNG、原始数值 NPZ、FEM 系数 TXT、损失 CSV 和 PNG、
注意力权重 PNG，以及运行记录 JSON。四个频率通道共用色标，不分别做峰值归一化，
显示不设阈值。已有重建不会被覆盖，重复运行请换一个输出目录。

## 其他数据与快速检查

```bash
python run_demo.py --list-cases
python run_demo.py --case DBLung_sim --output results/lung
python run_demo.py --case zebrafish_realworld --output results/zebrafish

# 短程检查运行流程；5 次迭代不代表完整重建质量。
python run_demo.py --epochs 5 --output results/smoke
python run_demo.py --device cpu --epochs 5 --output results/cpu-smoke

# 直接从预存结果生成图片，不重新优化。
python visualize.py

python -m unittest discover -s tests -v
```

共提供 8 组四频电压、圆形和胸腔两套敏感矩阵、投影矩阵及位置表。
数据与几何的对应关系见 [data/README.md](data/README.md)。
默认迭代设置沿用原始仿真入口，其他观测可通过 `--epochs` 和 `--learning-rate` 配置。

## 接入自己的测量

```python
from dataclasses import replace
from maip.data import load_case
from maip.reconstruction import Config, reconstruct

measurement = load_case("case2_sim")
# your_voltages 为 float32、形状 (104, 4)，测量顺序和电压约定需与算子一致。
# measurement = replace(measurement, voltages=your_voltages)
arrays, info = reconstruct(measurement, Config(), device="auto")
```

核心实现位于 `maip/model.py` 和 `maip/reconstruction.py`，保留原始的有效网络类、
初始化、两次敏感矩阵归一化以及推理计算。数据以 NPZ 无损压缩打包。
更换几何时需提供一致的敏感矩阵、投影矩阵和网格位置表。

原始输入文件未提供电导率真值，因此此示例报告电压拟合误差。
这些文件也未附物理频率标签，图片按原始列顺序标为 Frequency 1–4。
不同设备和 PyTorch 版本可能有浮点差异。

## 引用与许可证

使用本代码、数据或方法时，请引用对应论文：
DOI：[10.1109/JIOT.2025.3624228](https://doi.org/10.1109/JIOT.2025.3624228)。
完整引用和 BibTeX 见 [README.md](README.md#citation-and-license) 与 `citation.bib`，
GitHub 的 **Cite this repository** 入口也优先给出此论文。

代码和作者的数值输入数据采用 [MIT 许可证](LICENSE)。
