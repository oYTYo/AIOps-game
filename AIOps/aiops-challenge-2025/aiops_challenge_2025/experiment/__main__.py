import argparse
import datetime
from functools import partial
import importlib
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from aiops_challenge_2025 import config
from aiops_challenge_2025.data import DataHelper, fillna, resample_data
from aiops_challenge_2025.experiment import (
    DATA_INTERVAL,
    PREDICT_NUM,
    PREDICT_WINDOW,
    TARGET_KPIS,
    ChangeInfo,
    ModelInterface,
)


def dynamic_import(name: str):
    """对于形如 "module.name:obj" 的输入，首先加载 module.name ，其次获取其 obj 属性"""
    module_name, obj_name = name.split(":")
    return getattr(importlib.import_module(module_name), obj_name)


class Evaluator:

    def __init__(
        self, data_dirs: List[str], task_filename: str, output_dir: str, model_path: str
    ):
        self._data_helper = DataHelper(data_dirs=data_dirs)
        self._data_helper_label = DataHelper(
            data_dirs=data_dirs, feature_groups=[config.FEATURE_GROUP_KPI]
        )
        with open(task_filename, encoding="UTF-8") as obj:
            tasks = [ChangeInfo(**item) for item in json.load(obj)]
        self._tasks = sorted(tasks, key=lambda task: task.start_time)
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        self._model_path = model_path
        self._model: ModelInterface = None

    @property
    def model(self) -> ModelInterface:
        """待测模型接口"""
        if not self._model:
            self._model = dynamic_import(self._model_path)
        return self._model

    def _data_getter(self, cell_id: str, predict_start: datetime.datetime):
        data = self._data_helper.get(cell_id=cell_id)
        return {
            feature_group: item.loc[item.index < predict_start]
            for feature_group, item in data.items()
        }

    @staticmethod
    def _resample(data: pd.DataFrame, start_time: datetime.datetime):
        data = resample_data(
            data,
            start=start_time,
            end=start_time + PREDICT_WINDOW,
            interval=DATA_INTERVAL,
        ).iloc[:PREDICT_NUM]
        data, non_missing = fillna(data)
        return data.astype(float), non_missing

    def inference(self):
        """模型推理"""
        raise NotImplementedError

    def evaluate(self) -> float:
        """评估推理结果"""
        raise NotImplementedError


class PredictEvaluator(Evaluator):

    def _output_filename(self, task: ChangeInfo):
        return os.path.join(self._output_dir, f"{task.identifier}.csv")

    def inference(self):
        for task in self._tasks:
            if os.path.isfile(self._output_filename(task)):
                continue
            result = self.model.predict(
                task=task,
                data_getter=partial(self._data_getter, predict_start=task.start_time),
            )
            result.to_csv(self._output_filename(task), index=True)

    def evaluate(self) -> float:
        results = []
        for task in self._tasks:
            result = config.FEATURE_GROUP_KPI.reindex(
                pd.read_csv(self._output_filename(task))
            )
            result, _ = self._resample(result[TARGET_KPIS], start_time=task.start_time)
            label = self._data_helper_label.get(cell_id=task.cell_id)[
                config.FEATURE_GROUP_KPI
            ][TARGET_KPIS]
            label, non_missing = self._resample(label, start_time=task.start_time)
            result, label, non_missing = [
                item.values for item in [result, label, non_missing]
            ]
            divisor = np.abs(label) + np.abs(result)
            divisor = np.where(divisor > 0, divisor, 1)
            smape = 2 * np.abs(label - result) * non_missing / divisor
            column_smapes: np.ndarray = smape.sum(axis=0) / non_missing.sum(axis=0)
            results.append(
                [task.cell_id, task.start_time, smape.sum() / non_missing.sum()]
                + column_smapes.tolist()
            )
        results = pd.DataFrame(
            results,
            columns=["cell_id", "start_time", "all"] + TARGET_KPIS,
        )
        results.to_csv(os.path.join(self._output_dir, "smape.csv"), index=False)
        return results["all"].mean()


class DecisionEvaluator(Evaluator):

    @property
    def _output_filename(self):
        return os.path.join(self._output_dir, "decisions.json")

    def inference(self):
        result = {}
        for task in self._tasks:
            if not task.change:
                continue
            target = self._data_helper_label.get(cell_id=task.cell_id)[
                config.FEATURE_GROUP_KPI
            ]
            target, _ = self._resample(target, start_time=task.start_time)
            result[task.identifier] = self.model.decide(
                cell_id=task.cell_id,
                start_time=task.start_time,
                target=target,
                data_getter=partial(self._data_getter, predict_start=task.start_time),
            )
        with open(self._output_filename, "w") as obj:
            json.dump(result, obj, ensure_ascii=False, indent=2)

    def evaluate(self) -> float:
        with open(self._output_filename) as obj:
            # {ChangeInfo.identifier: {parameter: value}}
            model_result: Dict[str, Dict[str, float]] = json.load(obj)
        results = []
        for task in self._tasks:
            for parameter, value in task.change.items():
                model_value = model_result.get(task.identifier, {}).get(
                    parameter, np.nan
                )
                divisor = abs(value) + abs(model_value)
                if divisor == 0:
                    divisor = 1
                results.append(
                    (
                        task.cell_id,
                        task.start_time,
                        parameter,
                        2 * abs(value - model_value) / divisor,
                    )
                )
        results = pd.DataFrame(
            results, columns=["cell_id", "start_time", "parameter", "smape"]
        )
        results.to_csv(os.path.join(self._output_dir, "smape.csv"), index=False)
        return results["smape"].mean()


def _main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sample_data_dir = os.path.join(os.path.dirname(config.CURRENT_DIR), "sample")
    parser.add_argument(
        "-d",
        "--data-dir",
        action="append",
        default=[],
        required=True,
        dest="data_dirs",
        help=f"数据路径，目录结构形如 {sample_data_dir}。可指定多个。",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        dest="task",
        help=f'预测任务清单，格式形如 {os.path.join(sample_data_dir, "changes.json")}',
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="对模型输出打分。默认为调用模型并保存推理结果",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="naive_baseline:model",
        type=str,
        required=True,
        help="待测模型入口",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        type=str,
        required=False,
        help="输出目录",
    )
    parser.add_argument(
        "action",
        default="predict",
        type=str,
        choices=("predict", "decide"),
    )

    args = parser.parse_args()
    evaluator_cls = DecisionEvaluator if args.action == "decide" else PredictEvaluator
    evaluator = evaluator_cls(
        data_dirs=args.data_dirs,
        task_filename=args.task,
        output_dir=args.output_dir,
        model_path=args.model,
    )
    if args.evaluate:
        print("smape =", evaluator.evaluate())
    else:  # predict
        evaluator.inference()


if __name__ == "__main__":
    _main()
