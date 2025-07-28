import dataclasses
import datetime
from typing import Callable, Dict

import pandas as pd

from aiops_challenge_2025.config import FeatureGroup


# 数据采样间隔
DATA_INTERVAL = datetime.timedelta(minutes=15)
# 需要预测的数据点数量
PREDICT_NUM = 96
# 需要预测的窗口长度
PREDICT_WINDOW = 96 * DATA_INTERVAL

TARGET_KPIS = [
    "ho_succoutinterenbs1",
    "ho_succoutinterenbx2",
    "ho_succoutintraenb",
    "ho_attoutinterenbs1",
    "ho_attoutinterenbx2",
    "ho_attoutintraenb",
    "pdcp_upoctdl",
    "pdcp_upoctul",
    "rru_pdschprbassn",
    "rru_puschprbassn",
    "rrc_connmax",
    "rrc_connmean",
    "rrc_succconnestab",
    "rrc_attconnreestab",
    "rrc_attconnestab",
    "erab_nbrsuccestab",
    "erab_nbrattestab",
    "erab_nbrreqrelenb",
    "erab_nbrleft",
    "erab_nbrfailestab",
    "context_succinitalsetup",
    "pdcp_thrptimeul",
    "pdcp_thrptimedl",
    "pdcp_nbrpktdl",
    "pdcp_nbrpktul",
    "pdcp_uppkttotdelaydl",
    "pdcp_nbrpktlossdl",
    "pdcp_nbrpktlossul",
    "rru_pdcchcceavail",
    "rru_pdcchcceutil",
    "erab_nbrmeanestab_1",
    "erab_nbrhoinc",
    "pdcp_uplastttioctul",
    "pdcp_uplastttioctdl",
    "ho_succoutintrafreq",
    "ho_succoutinterfreq",
    "ho_succexecinc",
    "rrc_effectiveconnmean",
    "rrc_effectiveconnmax",
    "succ_conn_rate",
    "ho_succ_rate",
    "erab_nbrreqrelenb.1",
    "erab_nbrreqrelenb_normal.1",
]


def prev_quarter_time(timestamp: datetime.datetime) -> datetime.datetime:
    """
    对齐到不超过给定时间的整 15 分钟

    >>> prev_quarter_time(
    ...     datetime.datetime(year=2025, month=1, day=1, hour=0, minute=16, second=23)
    ... )
    datetime.datetime(2025, 1, 1, 0, 15)
    >>> prev_quarter_time(
    ...     datetime.datetime(year=2025, month=1, day=1, hour=0, minute=15)
    ... )
    datetime.datetime(2025, 1, 1, 0, 15)
    """
    return timestamp.replace(
        minute=15 * (timestamp.minute // 15), second=0, microsecond=0
    )


@dataclasses.dataclass
class ChangeInfo:

    cell_id: str = dataclasses.field(metadata={"help": "待预测的小区标识"})
    start_time: datetime.datetime = dataclasses.field(
        metadata={"help": "待预测的开始时刻，保证对齐到整 15 分钟"}
    )
    change: Dict[str, float] = dataclasses.field(
        default_factory=dict, metadata={"help": "变更后的参数"}
    )

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.datetime.fromisoformat(self.start_time)
        start_time = prev_quarter_time(self.start_time)
        if start_time < self.start_time:
            self.start_time = start_time + DATA_INTERVAL

    @property
    def identifier(self) -> str:
        """将调参小区与调参时刻编码为字符串"""
        return "_".join([self.cell_id, self.start_time.strftime("%Y%m%d%H%M%S")])


class ModelInterface:

    def predict(
        self,
        task: ChangeInfo,
        data_getter: Callable[[str], Dict[FeatureGroup, pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        预测给定小区的 KPI ，需要预测的数据点数量由 PREDICT_NUM 给出

        @param task 指定待预测的小区标识、预测的开始时刻、变更后的参数
        @param data_getter 获取给定小区的数据，返回空字典表示缺少数据。可多次调用以获取邻区数据。
        """
        raise NotImplementedError

    def decide(
        self,
        cell_id: str,
        start_time: datetime.datetime,
        target: pd.DataFrame,
        data_getter: Callable[[str], Dict[FeatureGroup, pd.DataFrame]],
    ) -> Dict[str, float]:
        """
        判断将给定小区的 KPI 调整到目标值需要调整的参数及其取值

        @param cell_id 待决策的小区标识
        @param start_time 调参的开始时刻
        @param target KPI 的变化目标
        @param data_getter 获取给定小区的数据，返回空字典表示缺少数据。可多次调用以获取邻区数据。
        """
        raise NotImplementedError
