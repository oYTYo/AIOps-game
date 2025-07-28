# 使用说明

## 命令行工具

```python
# 1. 安装依赖
pip install .
# 2.1 生成预测输出
# 执行`python -m aiops_challenge_2025.experiment -h`可以看到样例数据的绝对路径。
# `-m`参数用于指定`aiops_challenge_2025.experiment.ModelInterface`的实例，
# 冒号前将作为 import 的内容，冒号后将作为属性名。
python -m aiops_challenge_2025.experiment \
    -d aiops_challenge_2025/sample/ \
    -t aiops_challenge_2025/sample/changes.json \
    -m naive_baseline:model \
    -o output/predict \
    predict
# 2.2 评估预测结果
# 由于样例数据中大部分列没有数据，对应列在计算 smape 时会提示 RuntimeWarning。
python -m aiops_challenge_2025.experiment \
    -d aiops_challenge_2025/sample/ \
    -t aiops_challenge_2025/sample/changes.json \
    -m naive_baseline:model \
    -o output/predict \
    -e predict
# 3.1 生成决策输出
python -m aiops_challenge_2025.experiment \
    -d aiops_challenge_2025/sample/ \
    -t aiops_challenge_2025/sample/changes.json \
    -m naive_baseline:model \
    -o output/decide \
    decide
# 3.2 评估决策结果
python -m aiops_challenge_2025.experiment \
    -d aiops_challenge_2025/sample/ \
    -t aiops_challenge_2025/sample/changes.json \
    -m naive_baseline:model \
    -o output/decide \
    -e decide
```

## 样例模型

`aiops_challenge_2025.experiment`的自动评测功能需要选手继承[aiops_challenge_2025.experiment.ModelInterface](aiops_challenge_2025/experiment/__init__.py)并实现`predict`和`decide`方法。
[naive_baseline.py](naive_baseline.py) 提供了基于简单复制历史数据的样例实现。
样例模型的训练入口为`python -m naive_baseline -d aiops_challenge_2025/sample/`。
