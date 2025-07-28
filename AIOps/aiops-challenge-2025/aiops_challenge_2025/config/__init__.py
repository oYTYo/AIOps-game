import dataclasses
import datetime
import os
from typing import List

import pandas as pd
import yaml


@dataclasses.dataclass
class FeatureGroup:
    """一组特征"""

    data_id: str = dataclasses.field()
    time_column: str = dataclasses.field(default="time")
    sample_interval: datetime.timedelta = dataclasses.field(
        default=datetime.timedelta(minutes=15)
    )
    nominal: List[str] = dataclasses.field(
        default_factory=list, metadata={"help": "枚举类型字段"}
    )
    numeric: List[str] = dataclasses.field(
        default_factory=list, metadata={"help": "数值类型字段"}
    )

    def __post_init__(self):
        if isinstance(self.sample_interval, (float, int)):
            self.sample_interval = datetime.timedelta(seconds=self.sample_interval)
        elif isinstance(self.sample_interval, str):
            self.sample_interval = pd.to_timedelta(self.sample_interval)

    def __hash__(self) -> int:
        return hash(self.data_id)

    def __eq__(self, value) -> bool:
        return isinstance(value, FeatureGroup) and value.data_id == self.data_id

    @property
    def columns(self) -> List[str]:
        """数据字段清单"""
        return self.numeric + self.nominal

    @property
    def all_columns(self) -> List[str]:
        """包含时间的原始字段清单"""
        return [self.time_column] + self.columns

    @classmethod
    def load(cls, filename: str):
        with open(filename, encoding="UTF-8") as obj:
            data: dict = yaml.load(obj, yaml.SafeLoader)
        return cls(**data)

    def reindex(self, data: pd.DataFrame):
        """
        规范列

        - 解析日期列
        - 筛选出数据列，对于缺失的列添加空白列
        - 设置日期列为索引列
        """
        data = data.reindex(columns=self.all_columns)
        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data.set_index(self.time_column, drop=True, append=False, inplace=True)
        data.sort_index(inplace=True)
        return data


@dataclasses.dataclass(eq=False)
class RelationFeatureGroup(FeatureGroup):
    """邻区特征"""

    neighbour_column: str = dataclasses.field(default="cgi_nc")

    @property
    def all_columns(self) -> List[str]:
        """包含时间和邻区标识的原始字段清单"""
        return [self.time_column, self.neighbour_column] + self.columns


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_GROUP_EP, FEATURE_GROUP_KPI, FEATURE_GROUP_NRM = (
    FeatureGroup.load(os.path.join(CURRENT_DIR, f"{data_id}.yaml"))
    for data_id in ["EP", "KPI", "NRM"]
)
FEATURE_GROUP_MR, FEATURE_GROUP_PM_RELATION = (
    RelationFeatureGroup.load(os.path.join(CURRENT_DIR, f"{data_id}.yaml"))
    for data_id in ["MR", "PMRelation"]
)
