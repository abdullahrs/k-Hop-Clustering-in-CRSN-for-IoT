# pu_activity_model.py

from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Tuple


class ChannelState(Enum):
    """Enum for channel states"""

    IDLE = "IDLE"  # Channel is available (OFF state)
    BUSY = "BUSY"  # Channel is occupied by PU (ON state)


@dataclass
class PUActivityPattern:
    """Represents different PU activity patterns based on α and β values"""

    name: str
    alpha: float  # departure rate
    beta: float  # arrival rate
    description: str


class PUActivityModel:
    """Models Primary User activity on channels"""

    # Define the four cases of PU activity patterns as per paper
    ACTIVITY_PATTERNS = {
        "CASE1": PUActivityPattern(
            "CASE1",
            alpha=0.5,  # α ≤ 1
            beta=0.5,  # β ≤ 1
            description="Long ON periods followed by long OFF periods",
        ),
        "CASE2": PUActivityPattern(
            "CASE2",
            alpha=0.5,  # α ≤ 1
            beta=2.0,  # β > 1
            description="Long ON periods followed by short OFF periods",
        ),
        "CASE3": PUActivityPattern(
            "CASE3",
            alpha=2.0,  # α > 1
            beta=0.5,  # β ≤ 1
            description="Short ON periods followed by long OFF periods",
        ),
        "CASE4": PUActivityPattern(
            "CASE4",
            alpha=2.0,  # α > 1
            beta=2.0,  # β > 1
            description="Short ON periods followed by short OFF periods",
        ),
    }

    def __init__(self, alpha: float, beta: float):
        """
        Initialize PU activity model with given α and β values

        Args:
            alpha: PU departure rate (OFF→ON transition rate)
            beta: PU arrival rate (ON→OFF transition rate)
        """
        self.alpha = alpha
        self.beta = beta
        self.current_state = ChannelState.IDLE

    @property
    def p_on(self) -> float:
        """Calculate probability of channel being ON (occupied by PU)"""
        return self.beta / (self.alpha + self.beta)

    @property
    def p_off(self) -> float:
        """Calculate probability of channel being OFF (available)"""
        return self.alpha / (self.alpha + self.beta)

    def get_idle_duration(self) -> float:
        """Generate exponentially distributed idle duration"""
        return np.random.exponential(1 / self.alpha)

    def get_busy_duration(self) -> float:
        """Generate exponentially distributed busy duration"""
        return np.random.exponential(1 / self.beta)

    def update_state(
        self, current_time: float, last_transition_time: float
    ) -> Tuple[ChannelState, float]:
        """
        Update channel state based on exponential distribution

        Args:
            current_time: Current simulation time
            last_transition_time: Time of last state transition

        Returns:
            Tuple of (new_state, next_transition_time)
        """
        if self.current_state == ChannelState.IDLE:
            duration = self.get_idle_duration()
            if current_time - last_transition_time >= duration:
                self.current_state = ChannelState.BUSY
                return ChannelState.BUSY, current_time + self.get_busy_duration()
        else:
            duration = self.get_busy_duration()
            if current_time - last_transition_time >= duration:
                self.current_state = ChannelState.IDLE
                return ChannelState.IDLE, current_time + self.get_idle_duration()

        return self.current_state, last_transition_time + duration

    @classmethod
    def determine_activity_pattern(cls, alpha: float, beta: float) -> PUActivityPattern:
        """Determine which activity pattern case the given α and β values fall into"""
        if alpha <= 1:
            if beta <= 1:
                return cls.ACTIVITY_PATTERNS["CASE1"]
            else:
                return cls.ACTIVITY_PATTERNS["CASE2"]
        else:
            if beta <= 1:
                return cls.ACTIVITY_PATTERNS["CASE3"]
            else:
                return cls.ACTIVITY_PATTERNS["CASE4"]


class Channel:
    """Enhanced Channel class with PU activity modeling"""

    def __init__(self, id: int, alpha: float, beta: float):
        self.id = id
        self.pu_activity = PUActivityModel(alpha, beta)
        self.last_transition_time = 0
        self.current_time = 0
        self.quality = 0.0

    def __hash__(self):
        return self.id

    def update(self, simulation_time: float):
        """Update channel state based on PU activity"""
        self.current_time = simulation_time
        new_state, next_transition = self.pu_activity.update_state(
            self.current_time, self.last_transition_time
        )

        if next_transition != self.last_transition_time:
            self.last_transition_time = self.current_time

        return new_state

    @property
    def is_available(self) -> bool:
        """Check if channel is available for SUs"""
        return self.pu_activity.current_state == ChannelState.IDLE

    def get_channel_quality(self, epsilon: float = 1.5) -> float:
        """Calculate channel quality using equation from paper"""
        Pon = self.pu_activity.p_on
        D = 1 / self.pu_activity.alpha  # Average idle duration
        self.quality = (1 + np.log(epsilon * Pon)) * D
        return self.quality
