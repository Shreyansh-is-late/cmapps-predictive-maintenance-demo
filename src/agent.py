import json
from datetime import datetime

class TriageAgent:
    def __init__(self, warning_threshold=30, critical_threshold=15):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def evaluate_rul(self, unit_id, current_cycle, predicted_rul):
        """Evaluates the RUL and generates a work order if thresholds are breached."""
        if predicted_rul <= self.critical_threshold:
            severity = "CRITICAL"
            action = "Immediate machine shutdown and emergency maintenance."
        elif predicted_rul <= self.warning_threshold:
            severity = "WARNING"
            action = "Schedule routine maintenance within the next 5 cycles."
        else:
            severity = "NORMAL"
            action = "Continue standard operations."

        return self._generate_work_order(unit_id, current_cycle, predicted_rul, severity, action)

    def _generate_work_order(self, unit_id, current_cycle, rul, severity, action):
        """Constructs the JSON work order payload."""
        if severity == "NORMAL":
            return None
        
        work_order = {
            "work_order_id": f"WO-UNIT{int(unit_id)}-{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "unit_id": int(unit_id),
            "cycle_detected": int(current_cycle),
            "predicted_rul": round(float(rul), 2),
            "severity": severity,
            "recommended_action": action
        }
        return work_order

if __name__ == "__main__":
    # Quick test of the agent logic
    agent = TriageAgent()
    print("Testing Normal State (RUL 45):", agent.evaluate_rul(1, 100, 45.0))
    print("Testing Warning State (RUL 25):")
    print(json.dumps(agent.evaluate_rul(1, 120, 25.0), indent=2))