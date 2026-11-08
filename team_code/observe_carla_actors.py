# -*- coding: utf-8 -*-
"""Independent local CARLA ground-truth observer for failed-route analysis.

Run in a SECOND terminal during a local Bench2Drive evaluation. This process
never sends controls, never ticks the world, and never passes ground-truth data
to the SENSORS agent. Only use it for offline diagnostics, not model input.

The optional collision sensor adds a CARLA sensor actor to the simulation;
leave it OFF during official benchmark scoring and use only on debug reruns.
"""
import argparse
import json
import math
import queue
import time
from pathlib import Path

import carla


def _xyz(p):
    return [float(p.x), float(p.y), float(p.z)]


def _bbox(actor):
    b = actor.bounding_box
    t = actor.get_transform()
    return {
        'extent_xyz_m': _xyz(b.extent),
        'offset_xyz_m': _xyz(b.location),
        'rotation_rpy_deg': [float(b.rotation.roll), float(b.rotation.pitch), float(b.rotation.yaw)],
        'world_vertices_xyz_m': [_xyz(v) for v in b.get_world_vertices(t)],
    }


def _ego_local(ego_transform, location):
    ego = ego_transform.location
    yaw = math.radians(ego_transform.rotation.yaw)
    dx, dy = float(location.x - ego.x), float(location.y - ego.y)
    return [dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw)]


def _actor_info(actor, ego_transform):
    tf = actor.get_transform()
    v = actor.get_velocity()
    local = _ego_local(ego_transform, tf.location)
    return {
        'id': int(actor.id),
        'type_id': str(actor.type_id),
        'role_name': str(actor.attributes.get('role_name', '')),
        'location_xyz_m': _xyz(tf.location),
        'rotation_rpy_deg': [float(tf.rotation.roll), float(tf.rotation.pitch), float(tf.rotation.yaw)],
        'velocity_xyz_mps': _xyz(v),
        'speed_mps': float(math.hypot(v.x, v.y)),
        'ego_relative_center_forward_left_m': [float(local[0]), float(local[1])],
        'center_distance_m': float(math.hypot(local[0], local[1])),
        'bbox': _bbox(actor),
    }


def _find_ego(world, preferred_id=None):
    vehicles = list(world.get_actors().filter('vehicle.*'))
    if preferred_id is not None:
        matches = [a for a in vehicles if a.id == preferred_id]
        if matches:
            return matches[0]
    candidates = [a for a in vehicles if a.attributes.get('role_name', '').lower()
                  in ('hero', 'ego_vehicle')]
    return candidates[0] if len(candidates) == 1 else None


def main():
    p = argparse.ArgumentParser(description='Independent CARLA actor GT recorder (no model/control access).')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=10000)
    p.add_argument('--out', required=True, help='Output directory, independent of model diagnostics')
    p.add_argument('--radius', type=float, default=40.0)
    p.add_argument('--ego-id', type=int, default=None)
    p.add_argument('--collision-sensor', action='store_true',
                   help='Optional debug-only sensor: DO NOT enable for official scoring')
    p.add_argument('--connect-timeout', type=float, default=180.0)
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'observer_manifest.json').write_text(json.dumps({
        'format': 'carla_actor_observer_v1',
        'port': args.port,
        'ego_local_axes': 'forward positive X, left positive Y, via CARLA world yaw',
        'bbox': 'CARLA actor ground-truth bounding-box vertices (world XYZ)',
        'frame_alignment': 'CARLA world snapshot frame; not the agent step index',
        'collision_sensor': args.collision_sensor,
        'warning': 'Observer is external; no truth passed to SENSORS agent; synchronous tick frames may be skipped if observer is slow.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    client = carla.Client(args.host, args.port)
    deadline = time.monotonic() + args.connect_timeout
    world = None
    while time.monotonic() < deadline:
        try:
            client.set_timeout(2.0)
            world = client.get_world()
            print('[ActorObserver] Connected to CARLA port {}'.format(args.port), flush=True)
            break
        except RuntimeError:
            time.sleep(1)
    if world is None:
        raise RuntimeError('CARLA RPC not reachable at {}:{}'.format(args.host, args.port))

    collision_queue = queue.Queue()
    sensor = None
    sensor_parent_id = None
    last_frame = -1
    try:
        with (root / 'actor_frames.jsonl').open('w', encoding='utf-8', buffering=1) as frame_f, \
             (root / 'collision_events.jsonl').open('w', encoding='utf-8', buffering=1) as collision_f:
            print('[ActorObserver] Waiting for hero vehicle. Press Ctrl+C after collision.', flush=True)
            while True:
                try:
                    snapshot = world.wait_for_tick(seconds=3.0)
                except RuntimeError:
                    # World can be reloaded while a route starts or ends.
                    try:
                        world = client.get_world()
                    except RuntimeError:
                        time.sleep(0.5)
                    continue
                if snapshot is None or snapshot.frame == last_frame:
                    continue
                last_frame = int(snapshot.frame)
                ego = _find_ego(world, args.ego_id)
                if ego is None:
                    continue
                tf = ego.get_transform()
                if args.collision_sensor and (sensor is None or sensor_parent_id != ego.id):
                    if sensor is not None:
                        try:
                            sensor.stop()
                            sensor.destroy()
                        except RuntimeError:
                            pass
                    sensor = world.spawn_actor(
                        world.get_blueprint_library().find('sensor.other.collision'),
                        carla.Transform(), attach_to=ego)
                    sensor.listen(lambda event, ego_id=int(ego.id): collision_queue.put({
                        'carla_frame': int(event.frame),
                        'carla_elapsed_seconds': float(event.timestamp),
                        'ego_actor_id': ego_id,
                        'other_actor_id': int(event.other_actor.id),
                        'other_actor_type_id': str(event.other_actor.type_id),
                        'normal_impulse_xyz': _xyz(event.normal_impulse),
                    }))
                    sensor_parent_id = ego.id
                    print('[ActorObserver] Collision sensor attached to hero {}'.format(ego.id), flush=True)
                others = []
                for actor in world.get_actors().filter('vehicle.*'):
                    if actor.id == ego.id:
                        continue
                    d = actor.get_location().distance(tf.location)
                    if d <= args.radius:
                        others.append(_actor_info(actor, tf))
                others.sort(key=lambda a: a['center_distance_m'])
                frame_f.write(json.dumps({
                    'carla_frame': int(snapshot.frame),
                    'carla_elapsed_seconds': float(snapshot.timestamp.elapsed_seconds),
                    'ego': _actor_info(ego, tf),
                    'vehicles_within_radius': others,
                    'closest_vehicle_id': others[0]['id'] if others else None,
                    'closest_center_distance_m': others[0]['center_distance_m'] if others else None,
                }, ensure_ascii=False, allow_nan=False) + '\n')
                while not collision_queue.empty():
                    item = collision_queue.get_nowait()
                    collision_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print('[ActorObserver] Collision: {}'.format(item), flush=True)
    except KeyboardInterrupt:
        print('[ActorObserver] Stopping.', flush=True)
    finally:
        if sensor is not None:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass


if __name__ == '__main__':
    main()
