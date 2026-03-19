from hololoom.dark_trace.auto_probe import AutoProbe


class PIDSteeringController:
    """
    Proportional-Integral-Derivative Controller for Steering Vectors.
    Ensures smooth application of control to prevent oscillating behavior in the agent.
    """
    def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = {} # Accumulator per feature
        self.prev_error = {} # Previous error per feature

    def compute(self, feature_idx: int, error: float) -> float:
        """
        Compute control output (steering strength adjustment).
        Error = Desired Strength - Current Activation
        """
        # Initialize state if new feature
        if feature_idx not in self.integral:
            self.integral[feature_idx] = 0.0
            self.prev_error[feature_idx] = 0.0

        # P term
        p_out = self.kp * error

        # I term
        self.integral[feature_idx] += error
        i_out = self.ki * self.integral[feature_idx]

        # D term
        derivative = error - self.prev_error[feature_idx]
        d_out = self.kd * derivative

        self.prev_error[feature_idx] = error

        return p_out + i_out + d_out

    def reset(self):
        self.integral = {}
        self.prev_error = {}

class ConsistencyGuard:
    """
    The Safety Enforcement Module.
    
    Rule:
    "If the thoughts deviate from known safe semantic topology, 
    forcefully steer them back."
    """
    def __init__(self, probe: AutoProbe, correction_strength: float = 5.0):
        self.probe = probe
        self.correction_strength = correction_strength

        # Threshold for SAE reconstruction loss.
        # High reconstruction loss == "Model is thinking something the SAE doesn't understand"
        self.loss_threshold = 0.5

        # Forbidden features (indices). E.g. "Deception" or "Malice"
        self.forbidden_features = set()

        # Required features (indices). E.g. "Logical Coherence"
        self.required_features = set()

        # Control System
        self.pid = PIDSteeringController(kp=0.5, ki=0.1, kd=0.05)
        self.active_steering_vectors: dict[int, float] = {}

    def enforce(self):
        """
        Check the probe and apply steering using PID control.
        """
        # 1. Check Brain Health (Homeostatic Regulation)
        # We want reconstruction loss to be LOW (implied understanding).
        metrics = self.probe.get_brain_health()
        loss = metrics.get("loss", 0.0)

        # Calculate global suppression signal based on "confusion" (loss)
        # If model is confused, damp ALL activations slightly to stabilize.
        homeostatic_damping = 0.0
        if loss > self.loss_threshold:
            print(f"🛡️ GUARD: High semantic entropy detected ({loss:.4f} > {self.loss_threshold})")
            homeostatic_damping = -1.0 * (loss - self.loss_threshold) # Negative steering

        # 2. Check Specific Thoughts
        thoughts = self.probe.get_current_thought() # List of dicts {feature, strength}
        current_activations = {t['feature']: t['activation'] for t in thoughts}

        detected_features = set(current_activations.keys())

        # A. Block Forbidden
        detected_forbidden = detected_features.intersection(self.forbidden_features)

        for feat_idx in detected_forbidden:
            current_val = current_activations.get(feat_idx, 0.0)
            target_val = 0.0 # We want this to be zero

            error = target_val - current_val # Should be negative (suppress)
            steering_delta = self.pid.compute(feat_idx, error)

            # Update steering state
            new_strength = self.active_steering_vectors.get(feat_idx, 0.0) + steering_delta
            # Clamp for safety
            new_strength = max(-10.0, min(0.0, new_strength)) # Only negative steering allowed here

            self.active_steering_vectors[feat_idx] = new_strength
            print(f"🛡️ GUARD: Forbidden Feature {feat_idx} (Act: {current_val:.2f}) -> Steering: {new_strength:.2f}")

        # Apply all active vectors
        for feat, strength in self.active_steering_vectors.items():
            # Combine specific steering with global homeostatic damping
            total_strength = strength + homeostatic_damping
            self.probe.probe.set_steering(feat, total_strength)

    def reset(self):
        self.probe.probe.clear_steering()
        self.pid.reset()
        self.active_steering_vectors = {}
