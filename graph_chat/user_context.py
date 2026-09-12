"""
    乘客信息上下文
"""
from dataclasses import dataclass


@dataclass
class UserContext:
    passenger_id: str