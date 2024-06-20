{
  "hcp_profile": {
    "name": "Dr. John Doe",
    "specialty": "Cardiology",
    "location": "New York, NY",
    "preferred_communication_times": "Afternoons",
    "previous_interactions": ["Email - Positive response", "Call - No response"],
    "engagement_history": ["High email open rates", "Low call engagement"]
  },
  "strategic_imperatives": [
    {
      "imperative": "Increase awareness of new heart medication",
      "key_messages": [
        "Clinically proven to reduce heart attacks by 30%",
        "Minimal side effects compared to existing treatments"
      ]
    },
    {
      "imperative": "Highlight patient benefits",
      "key_messages": [
        "Improves quality of life",
        "Easily integrated into existing treatment plans"
      ]
    }
  ],
  "channel_performance": {
    "email": {"historical_effectiveness": "High", "recent_performance": "Good"},
    "call": {"historical_effectiveness": "Medium", "recent_performance": "Poor"},
    "samples": {"historical_effectiveness": "Medium", "recent_performance": "Medium"},
    "lunch_and_learn": {"historical_effectiveness": "High", "recent_performance": "High"}
  },
  "model_outputs": {
    "predicted_rx_increase": {
      "email": 15,
      "call": 5,
      "samples": 10,
      "lunch_and_learn": 20
    },
    "shap_values": {
      "email": {"specialty": 0.1, "location": 0.05, "previous_interactions": 0.2, "engagement_history": 0.15},
      "call": {"specialty": 0.05, "location": 0.02, "previous_interactions": 0.1, "engagement_history": 0.05},
      "samples": {"specialty": 0.08, "location": 0.03, "previous_interactions": 0.15, "engagement_history": 0.1},
      "lunch_and_learn": {"specialty": 0.12, "location": 0.07, "previous_interactions": 0.25, "engagement_history": 0.2}
    }
  }
}
