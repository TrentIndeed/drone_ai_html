from flask import Flask, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Constants (mirrored from JavaScript) ---
LEAN = { 'maxPitch': math.radians(22) }
SWEEP_HOLD = 3.0
SWEEP_TRANSIT = 0.9
SWEEP_ANGLE_DEG = 40
HALF = 900 * 0.48 # TERRAIN_SIZE * 0.48

# --- Global state for the search FSM ---
search_state = {
    'inited': False,
    'centerYaw': 0,
    'side': 1,
    'timer': 0,
    'phase': 'hold',
    'seekDir': None
}

# Global state for smoothed controls to prevent jitter
control_state = {
    'last_yaw': 0.0,
    'last_throttle': 0.0,
    'last_pitch': 0.0
}

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def lerp_angle(a, b, t):
    d = wrap_angle(b - a)
    return a + d * clamp(t, 0, 1)

def steer_avoid(pos, vel, obstacles):
    ahead_pos = {
        'x': pos['x'] + vel['x'] * 0.1, # Look ahead based on velocity
        'y': pos['y'],
        'z': pos['z'] + vel['z'] * 0.1
    }
    push = {'x': 0, 'y': 0, 'z': 0}

    # Obstacle avoidance
    for o in obstacles:
        d2 = (ahead_pos['x'] - o['pos']['x'])**2 + (ahead_pos['z'] - o['pos']['z'])**2
        r = o['radius'] + 1.8
        if d2 < r*r:
            # Add a force pushing away from the obstacle
            push_vec_x = ahead_pos['x'] - o['pos']['x']
            push_vec_z = ahead_pos['z'] - o['pos']['z']
            dist = math.sqrt(d2)
            if dist > 1e-6:
                push['x'] += (push_vec_x / dist) * (1 - dist / r)
                push['z'] += (push_vec_z / dist) * (1 - dist / r)

    # Boundary avoidance
    margin = 28
    dx = HALF - abs(pos['x'])
    dz = HALF - abs(pos['z'])
    fx, fz = 0, 0
    if dx < margin:
        fx = math.copysign(1, pos['x']) * (1 - dx / margin)
    if dz < margin:
        fz = math.copysign(1, pos['z']) * (1 - dz / margin)
    
    push['x'] -= fx * 2.0
    push['z'] -= fz * 2.0

    return push

@app.route('/get_controls', methods=['POST'])
def get_controls():
    global search_state
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        drone = data.get('drone')
        locked_target = data.get('lockedTarget')
        terrain = data.get('terrain')
        obstacles = data.get('obstacles', [])
        dt = data.get('dt')

        if not all([drone, terrain, dt is not None]):
            app.logger.error(f"Incomplete data received")
            return jsonify({"error": "Incomplete data provided"}), 400

        # --- Direction Finding ---
        raw_dir = None
        is_hunting = locked_target is not None
        # A "new hunt" starts on the frame we have a target, but the search FSM hasn't been reset yet.
        is_newly_hunting = is_hunting and search_state.get('inited', False)

        if is_hunting:
            # --- HUNTING MODE ---
            search_state['inited'] = False # Stop the structured search pattern
            
            to_t_x = locked_target['position']['x'] - drone['position']['x']
            to_t_z = locked_target['position']['z'] - drone['position']['z']
            
            lead_x = locked_target['velocity']['x'] * 0.6
            lead_z = locked_target['velocity']['z'] * 0.6

            final_dir_x = to_t_x + lead_x
            final_dir_z = to_t_z + lead_z

            len_sq = final_dir_x**2 + final_dir_z**2
            if len_sq > 1e-6:
                inv_len = 1 / math.sqrt(len_sq)
                raw_dir = {'x': final_dir_x * inv_len, 'z': final_dir_z * inv_len}
            else:
                raw_dir = {'x': math.cos(drone['yaw']), 'z': math.sin(drone['yaw'])}
        else:
            # --- SEARCHING MODE ---
            if not search_state.get('inited', False):
                search_state['inited'] = True
                search_state['centerYaw'] = drone['yaw']
                search_state['side'] = 1
                search_state['timer'] = SWEEP_HOLD
                search_state['phase'] = 'hold'
            
            if search_state['phase'] == 'transit':
                cyaw = math.atan2(-drone['position']['z'], -drone['position']['x'])
                search_state['centerYaw'] = lerp_angle(search_state['centerYaw'], cyaw, 1 - math.exp(-dt * 0.25))
            
            search_state['timer'] -= dt
            if search_state['timer'] <= 0:
                if search_state['phase'] == 'hold':
                    search_state['side'] *= -1
                    search_state['phase'] = 'transit'
                    search_state['timer'] = SWEEP_TRANSIT
                else:
                    search_state['phase'] = 'hold'
                    search_state['timer'] = SWEEP_HOLD
            
            amp = math.radians(SWEEP_ANGLE_DEG)
            yaw_t = wrap_angle(search_state['centerYaw'] + search_state['side'] * amp)
            raw_dir = {'x': math.cos(yaw_t), 'z': math.sin(yaw_t)}

        # --- Unified Steering & Smoothing ---
        drone_status = 'HUNTING' if is_hunting else 'SEARCHING'
        
        if search_state.get('seekDir') is None:
            search_state['seekDir'] = {'x': math.cos(drone['yaw']), 'z': math.sin(drone['yaw'])}
        
        # If this is the first frame of a hunt, snap the direction immediately.
        # Otherwise, smoothly interpolate to the new direction.
        seek_dir = search_state['seekDir'].copy() # Work with a copy
        if is_newly_hunting:
            seek_dir = raw_dir
        else:
            smoothing_gain = 8.0 if is_hunting else 2.2
            a = 1 - math.exp(-dt * smoothing_gain)
            seek_dir['x'] += (raw_dir['x'] - seek_dir['x']) * a
            seek_dir['z'] += (raw_dir['z'] - seek_dir['z']) * a

        # Normalize and update the final desired direction
        seek_len = math.sqrt(seek_dir['x']**2 + seek_dir['z']**2)
        if seek_len > 1e-6:
            seek_dir['x'] /= seek_len
            seek_dir['z'] /= seek_len
        
        search_state['seekDir'] = seek_dir
        desired_dir = seek_dir
        
        # --- Avoidance ---
        avoid_vec = steer_avoid(drone['position'], desired_dir, obstacles)
        desired_dir['x'] += avoid_vec['x']
        desired_dir['z'] += avoid_vec['z']
        
        dir_len = math.sqrt(desired_dir['x']**2 + desired_dir['z']**2)
        if dir_len > 1e-6:
            desired_dir['x'] /= dir_len
            desired_dir['z'] /= dir_len

        # --- AI CONTROLS CALCULATION ---
        desired_yaw = math.atan2(desired_dir['z'], desired_dir['x'])
        yaw_error = wrap_angle(desired_yaw - drone['yaw'])
        commanded_yaw_rate = clamp(yaw_error / max(dt, 1e-5), -drone['maxTurn'], drone['maxTurn'])
        hud_yaw_raw = commanded_yaw_rate / drone['maxTurn']

        ground_ref = max(terrain['height_at_drone'], terrain['ahead_max_height'])
        target_y = ground_ref + drone['targetAlt']
        climb_err = target_y - drone['position']['y']
        vertical_vel = drone['velocity']['y']
        damping = 0.9 * vertical_vel
        altitude_throttle = clamp((climb_err - damping) * 0.2, -1, 1)

        fwd_x = math.cos(drone['yaw'])
        fwd_z = math.sin(drone['yaw'])
        forward_speed = drone['velocity']['x'] * fwd_x + drone['velocity']['z'] * fwd_z
        accel_err = drone['speed'] - forward_speed
        power_command = clamp(accel_err * 0.1, -1, 1)

        hud_pitch_raw = power_command
        hud_throttle_raw = clamp(altitude_throttle + power_command * 0.5, -1, 1)

        # --- SMOOTHING ---
        smoothing_factor = 1.0 - math.exp(-dt * 10.0)
        
        smoothed_yaw = lerp_angle(control_state['last_yaw'], hud_yaw_raw, smoothing_factor)
        smoothed_throttle = control_state['last_throttle'] + (hud_throttle_raw - control_state['last_throttle']) * smoothing_factor
        smoothed_pitch = control_state['last_pitch'] + (hud_pitch_raw - control_state['last_pitch']) * smoothing_factor

        control_state['last_yaw'] = smoothed_yaw
        control_state['last_throttle'] = smoothed_throttle
        control_state['last_pitch'] = smoothed_pitch

        return jsonify({
            'controls': { 'yaw': smoothed_yaw, 'throttle': smoothed_throttle, 'pitch': smoothed_pitch },
            'searchState': search_state,
            'droneStatus': drone_status
        })

    except Exception as e:
        app.logger.error(f"An error occurred in /get_controls: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500


import logging
from waitress import serve

if __name__ == '__main__':
    # Reduce the default logging for every request to keep the console clean
    log = logging.getLogger('waitress')
    log.setLevel(logging.INFO)
    
    # Use the production-ready Waitress server instead of Flask's built-in one
    # This is multi-threaded and can handle many more requests per second
    serve(app, host='0.0.0.0', port=5000)
