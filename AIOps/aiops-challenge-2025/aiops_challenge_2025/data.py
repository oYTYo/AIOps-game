import datetime
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from aiops_challenge_2025.config import (
    FeatureGroup,
    FEATURE_GROUP_KPI,
    FEATURE_GROUP_EP,
    FEATURE_GROUP_NRM,
    FEATURE_GROUP_MR,
    FEATURE_GROUP_PM_RELATION,
)


def check_datetime_index(data: pd.DataFrame):
    """确保数据的索引是时间"""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("数据必须以 DatetimeIndex 为索引")


def resample_data(
    data: pd.DataFrame,
    start: datetime.datetime,
    end: datetime.datetime,
    interval: datetime.timedelta = datetime.timedelta(minutes=15),
) -> pd.DataFrame:
    """按指定时间间隔重新采样数据"""
    check_datetime_index(data)
    data = data.loc[(data.index >= start) & (data.index < end + interval)].copy()
    if data.index.min() > start:
        data.loc[start] = None
    if data.index.max() < end:
        data.loc[end] = None
    length = int((end - start) / interval) + 1
    return data.resample(interval, origin=start).last().iloc[:length]


def fillna(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """使用最近的历史数据填充或填0"""
    non_missing = pd.notna(data)
    # 时间戳表示区间的左端点
    data = data.ffill(inplace=False)
    data.fillna(0, inplace=True)
    return data, non_missing


class DataHelper:
    """加载数据的工具类"""

    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        feature_groups: Optional[List[FeatureGroup]] = None,
    ):
        self._data_dirs = (
            [os.path.normpath(data_dirs)]
            if isinstance(data_dirs, str)
            else [os.path.normpath(data_dir) for data_dir in data_dirs]
        )
        self._feature_groups = feature_groups or [
            FEATURE_GROUP_KPI,
            FEATURE_GROUP_EP,
            FEATURE_GROUP_NRM,
            FEATURE_GROUP_MR,
            FEATURE_GROUP_PM_RELATION,
        ]
        self._cells: Set[str] = set()
        for data_dir in self._data_dirs:
            self._cells.update(self.list_cells(data_dir))

    @staticmethod
    def list_cells(data_dir: str, cell_list_filename: str = "cells.csv") -> Set[str]:
        cells = set()
        for cell_id in pd.read_csv(
            os.path.join(data_dir, cell_list_filename),
            names=["cell_id"],
            index_col=False,
        )["cell_id"].unique():
            path = os.path.normpath(os.path.join(data_dir, cell_id))
            if cell_id and path.startswith(data_dir) and os.path.isdir(path):
                cells.add(cell_id)
            else:
                raise ValueError("Invalid cell id")
        return cells

    @staticmethod
    def load_datafile(data_dir: str, feature_group: FeatureGroup) -> pd.DataFrame:
        """
        加载数据为 DataFrame 并规范列

        - 解析日期列
        - 筛选出数据列，对于缺失的列添加空白列
        - 设置日期列为索引列
        """
        filename = os.path.join(data_dir, f"{feature_group.data_id}.csv")
        if not os.path.isfile(filename):
            return pd.DataFrame(
                [],
                index=pd.DatetimeIndex([], name=feature_group.time_column),
                columns=feature_group.columns,
                dtype=str,
            )
        data = pd.read_csv(filename, dtype=str)
        data.drop_duplicates([feature_group.time_column], keep="first", inplace=True)
        return feature_group.reindex(data)

    @property
    def cells(self) -> Set[str]:
        """全部小区标识"""
        return self._cells

    def get(self, cell_id: str) -> Dict[FeatureGroup, pd.DataFrame]:
        """加载小区数据"""
        data: Dict[FeatureGroup, pd.DataFrame] = {}
        for data_dir in self._data_dirs:
            data_dir = os.path.join(data_dir, cell_id)
            if not os.path.isdir(data_dir):
                continue
            for feature_group in self._feature_groups:
                item = self.load_datafile(
                    data_dir=data_dir, feature_group=feature_group
                )
                if feature_group not in data:
                    data[feature_group] = item
                else:
                    data[feature_group] = pd.concat(
                        [data[feature_group], item], axis="index"
                    )
        return data
