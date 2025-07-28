import argparse
import datetime
import os
from typing import Callable, Dict, List

import pandas as pd

from aiops_challenge_2025 import config
from aiops_challenge_2025.data import DataHelper, fillna, resample_data
from aiops_challenge_2025.experiment import (
    DATA_INTERVAL,
    PREDICT_NUM,
    PREDICT_WINDOW,
    ModelInterface,
    ChangeInfo,
    prev_quarter_time,
)


class NaiveImplementation(ModelInterface):

    def fit(self, data_dirs: List[str]):
        data_helper = DataHelper(
            data_dirs=data_dirs,
            # 这里只加载小区级数据。如果考虑邻区数据，则需在列表中加入对应配置或不指定 feature_groups
            feature_groups=[
                config.FEATURE_GROUP_KPI,
                config.FEATURE_GROUP_EP,
                config.FEATURE_GROUP_NRM,
            ],
        )
        print(data_helper.cells)
        for cell_id in data_helper.cells:
            for feature_group, data in data_helper.get(cell_id).items():
                print(cell_id, feature_group.data_id, type(data))
            break

    def predict(
        self,
        task: ChangeInfo,
        data_getter: Callable[[str], Dict[config.FeatureGroup, pd.DataFrame]],
    ) -> pd.DataFrame:
        data = data_getter(task.cell_id)
        for item in data.values():
            assert (item.index < task.start_time).all(), "待预测数据泄露！"
        item = data[config.FEATURE_GROUP_KPI]
        end_time = task.start_time - DATA_INTERVAL
        start_time = min(prev_quarter_time(item.index.min()), end_time - PREDICT_WINDOW)
        item = resample_data(
            item.astype(float), start=start_time, end=end_time, interval=DATA_INTERVAL
        )
        item, non_missing = fillna(item)
        ret = item.iloc[-PREDICT_NUM:]
        ret.index = (
            end_time + pd.RangeIndex(start=1, stop=PREDICT_NUM + 1) * DATA_INTERVAL
        )
        ret.index.name = config.FEATURE_GROUP_KPI.time_column
        return ret

    def decide(
        self,
        cell_id: str,
        start_time: datetime.datetime,
        target: pd.DataFrame,
        data_getter: Callable[[str], Dict[config.FeatureGroup, pd.DataFrame]],
    ) -> Dict[str, float]:
        data = data_getter(cell_id)
        for item in data.values():
            assert (item.index < start_time).all(), "数据泄露！"
        item = data[config.FEATURE_GROUP_NRM].astype(float)
        row = item.ffill().iloc[-1]
        return dict(zip(row.index, row.values))


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

model = NaiveImplementation()


def _main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    model.fit(args.data_dirs)


if __name__ == "__main__":
    _main()
