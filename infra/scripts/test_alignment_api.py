"""Quick test of ActionObservation API fix."""
from hololoom.alignment.deception_detection import ActionObservation, create_detector

# Test ActionObservation creation with corrected API
action_obs = ActionObservation(
    action_id='test_123',
    description='Test action',
    claimed_goals=['helpful']
)
print(f'ActionObservation created: {action_obs.action_id}')

# Test detector
detector = create_detector()
detector.goal_tracker.observe_action(action_obs)
print('Action observed successfully')
print('ALL ALIGNMENT API TESTS PASSED!')
