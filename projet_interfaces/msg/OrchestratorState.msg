uint8 STATE_IDLE = 0
uint8 STATE_INITIALIZING = 1
uint8 STATE_LISTENING = 2
uint8 STATE_ACKNOWLEDGING = 3
uint8 STATE_DETECTING = 4
uint8 STATE_APPROACHING = 5
uint8 STATE_GRASPING = 6
uint8 STATE_RETRACTING = 7
uint8 STATE_DELIVERING = 8
uint8 STATE_WAITING_HANDOVER = 9
uint8 STATE_RELEASING = 10
uint8 STATE_RECOVERING = 11
uint8 STATE_ERROR = 254
uint8 STATE_EMERGENCY_STOP = 255
std_msgs/Header header
uint8 current_state
uint8 previous_state
string current_step_name
string target_object_id
uint8 retry_count
ErrorCode last_error
