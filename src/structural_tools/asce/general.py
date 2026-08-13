from enum import Enum


class RiskCategory(Enum):
    I = "I"  # noqa: E741
    II = "II"
    III = "III"
    IV = "IV"

    @property
    def severity(self):
        order = {
            RiskCategory.I: 1,
            RiskCategory.II: 2,
            RiskCategory.III: 3,
            RiskCategory.IV: 4,
        }
        return order[self]

    def __lt__(self, other):
        return self.severity < other.severity
