"""独立 Piper 回零：指定 CAN 通道，MOVE_J 回到六关节零角度并保持使能。"""
from __future__ import annotations

import argparse
import math
import re
import signal
import time

from ..core.robot import piper as base


DEFAULT_JOINT_LIMIT_TOLERANCE_DEG = 5.0


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        '--channel', required=True,
        help='the ONE arm to home, e.g. can_leader1 or can_follower1',
    )
    parser.add_argument('--interface', default='socketcan')
    parser.add_argument('--firmware', choices=('auto','default','v183','v188','v189'), default='auto')
    parser.add_argument('--speed-percent', type=base.percentage, default=10)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--feedback-timeout', type=float, default=3.0)
    parser.add_argument('--stale-timeout', type=float, default=0.5,
                        help='stop if joint feedback is older than this many seconds')
    parser.add_argument('--joint-limit-tolerance-deg', type=float,
                        default=DEFAULT_JOINT_LIMIT_TOLERANCE_DEG,
                        help='feedback-only allowance beyond nominal joint limits; command limits stay enabled')
    parser.add_argument('--tolerance-deg', type=float, default=2.0)
    base.add_execute_argument(parser)
    return parser


def validate_args(args):
    if not args.channel.strip():
        raise ValueError('channel must not be empty')
    for name in ('timeout','feedback_timeout','tolerance_deg','stale_timeout'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if (not math.isfinite(args.joint_limit_tolerance_deg)
            or not 0 <= args.joint_limit_tolerance_deg <= 5):
        raise ValueError('joint-limit-tolerance-deg must be finite and in [0, 5]')
    if args.tolerance_deg > 3:
        raise ValueError('tolerance-deg must be at most 3')


def connect_selected_arm(args):
    create_config, factory, model, fw = base._load_sdk()
    variants = {'default':fw.DEFAULT,'v183':fw.V183,'v188':fw.V188,'v189':fw.V189}
    def config_for(version):
        return create_config(robot=model.PIPER, firmeware_version=version,
                             interface=args.interface, channel=args.channel)
    firmware = args.firmware
    if firmware == 'auto':
        probe = factory.create_arm(config_for(fw.DEFAULT))
        try:
            probe.connect()
            deadline = time.monotonic() + args.feedback_timeout
            version = None
            while time.monotonic() < deadline:
                message = probe.get_firmware()
                if message is not None:
                    version = message['software_version']
                    break
                time.sleep(.05)
            if version is None:
                raise RuntimeError(f'No firmware feedback on {args.channel}; specify --firmware if needed')
            match = re.search(r'V(\d+)\.(\d+)-(\d+)',version)
            if match is None:
                raise RuntimeError(f'Unrecognized firmware version: {version}')
            numbers = tuple(int(x) for x in match.groups())
            firmware = ('v189' if numbers >= (1,8,9) else
                        'v188' if numbers >= (1,8,8) else
                        'v183' if numbers >= (1,8,3) else 'default')
            print(f'{args.channel}: detected {version}; using {firmware}')
        finally:
            base.disconnect_quietly(probe)
    config = config_for(variants[firmware])
    robot = factory.create_arm(config)
    try:
        robot.connect()
        robot.set_joint_limits_enabled(True)
    except BaseException:
        base.disconnect_quietly(robot)
        raise
    return robot, config


class IncompleteJointFeedback(RuntimeError):
    """A joint pair has not yet provided fresh feedback."""


def _require_complete_joint_feedback(robot, feedback_timeout):
    parser = getattr(robot, '_parser', None)
    if parser is None:
        return
    now = time.time()
    for name, label in (
        ('joint_12', 'J1/J2'), ('joint_34', 'J3/J4'), ('joint_56', 'J5/J6')
    ):
        pair = getattr(parser, name, None)
        timestamp = 0.0 if pair is None else float(getattr(pair, 'timestamp', 0.0))
        age = now - timestamp
        if not math.isfinite(timestamp) or timestamp <= 0 or age > feedback_timeout or age < -1:
            raise IncompleteJointFeedback(
                f'{label} joint feedback is missing or stale'
            )


def wait_complete_joints(robot, config, timeout, stale_timeout,
                         joint_limit_tolerance_deg=DEFAULT_JOINT_LIMIT_TOLERANCE_DEG):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return read_joints(robot, config, stale_timeout, joint_limit_tolerance_deg)
        except IncompleteJointFeedback as error:
            last_error = error
            time.sleep(.02)
    raise TimeoutError(
        f'Complete six-joint feedback did not arrive within {timeout:g} s: {last_error}'
    )


def read_joints(robot, config, feedback_timeout, joint_limit_tolerance_deg=DEFAULT_JOINT_LIMIT_TOLERANCE_DEG):
    message = robot.get_joint_angles()
    if message is None:
        raise IncompleteJointFeedback('Joint feedback is missing')
    _require_complete_joint_feedback(robot, feedback_timeout)
    values = list(message.msg)
    stamp = float(message.timestamp)
    age = time.time() - stamp
    if not math.isfinite(stamp) or stamp <= 0 or age > feedback_timeout or age < -1:
        raise IncompleteJointFeedback('Joint feedback is stale')
    return validate_joint_values(values, config, joint_limit_tolerance_deg)


def validate_joint_values(values, config, joint_limit_tolerance_deg=DEFAULT_JOINT_LIMIT_TOLERANCE_DEG):
    """Validate a six-joint snapshot without assuming a particular CAN frame source."""
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise RuntimeError('Invalid joint feedback; stopping homing')
    tolerance = math.radians(joint_limit_tolerance_deg)
    for i, value in enumerate(values,1):
        lower, upper = config['joint_limits'][f'joint{i}']
        if value < lower-tolerance-1e-6 or value > upper+tolerance+1e-6:
            raise RuntimeError(
                f'J{i} feedback={math.degrees(value):+.3f} deg outside nominal '
                f'[{math.degrees(lower):.3f}, {math.degrees(upper):.3f}] deg '
                f'with {joint_limit_tolerance_deg:g} deg feedback tolerance'
            )
    # Keep measured angles intact: clipping could falsely report a completed home.
    return list(values)


def bounded_hold(joints, config):
    return [min(config['joint_limits'][f'joint{i}'][1],
                max(config['joint_limits'][f'joint{i}'][0],value))
            for i,value in enumerate(joints,1)]



def read_gripper_width(gripper, feedback_timeout):
    message = gripper.get_gripper_status()
    if message is None or message.msg.mode != 'width':
        raise RuntimeError('Gripper width feedback is missing; stopping homing')
    width = float(message.msg.value)
    stamp = float(message.timestamp)
    age = time.time() - stamp
    if not math.isfinite(width) or not -0.002 <= width <= 0.152:
        raise RuntimeError('Invalid gripper width feedback; stopping homing')
    if not math.isfinite(stamp) or stamp <= 0 or age > feedback_timeout or age < -1:
        raise RuntimeError('Gripper feedback is stale; stopping homing')
    return max(0.0, width)


def run_home(robot, config, args, stopped, *, gripper=None, gripper_force=1.0, gripper_speed_mm_s=10.0):
    joints = read_joints(robot,config,args.stale_timeout,args.joint_limit_tolerance_deg)
    gripper_command = None if gripper is None else read_gripper_width(gripper, args.stale_timeout)
    if gripper is not None and (not math.isfinite(gripper_speed_mm_s) or not 0 < gripper_speed_mm_s <= 50):
        raise ValueError('gripper_speed_mm_s must be in (0, 50]')
    base.ensure_arm_status_normal(robot)
    if stopped():
        raise InterruptedError('Homing interrupted before enable')
    # Select speed BEFORE motion mode, because speed writes alter mode bits.
    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_speed_percent(args.speed_percent)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(bounded_hold(joints,config))
    enable_deadline = time.monotonic() + 5.0
    while True:
        if stopped():
            raise InterruptedError('Homing interrupted while enabling')
        robot.enable()
        states = robot.get_joints_enable_status_list()
        if len(states) == 6 and all(states):
            break
        if time.monotonic() >= enable_deadline:
            raise TimeoutError('Enable was not acknowledged within 5 seconds')
        time.sleep(.05)
    if stopped():
        raise InterruptedError('Homing interrupted before target')
    base.ensure_arm_status_normal(robot)
    read_joints(robot,config,args.stale_timeout,args.joint_limit_tolerance_deg)
    print(f'HOMING {args.channel}: MOVE_J at {args.speed_percent}%; target=[0,0,0,0,0,0] deg; Ctrl+C E-stops the active arm')
    robot.move_j([0.0]*6)
    started = time.monotonic()
    last_gripper_tick = started
    settled_since = None
    last_print = -math.inf
    last_progress_at = started
    initial_joints = read_joints(
        robot, config, args.stale_timeout, args.joint_limit_tolerance_deg
    )
    best_total_error_deg = sum(math.degrees(abs(value)) for value in initial_joints)
    while time.monotonic()-started < args.timeout:
        if stopped():
            raise InterruptedError('Homing interrupted')
        now = time.monotonic()
        joints = read_joints(robot,config,args.stale_timeout,args.joint_limit_tolerance_deg)
        gripper_width = None
        if gripper is not None:
            gripper_command = max(
                0.0, gripper_command - gripper_speed_mm_s / 1000.0 * max(0.0, now - last_gripper_tick)
            )
            last_gripper_tick = now
            gripper.move_gripper_m(gripper_command, gripper_force)
            gripper_width = read_gripper_width(gripper, args.stale_timeout)
        status = robot.get_arm_status()
        if status is None or base.value_as_int(status.msg.arm_status) != 0:
            raise RuntimeError('Arm status missing or abnormal; stopping homing')
        error = math.degrees(max(abs(x) for x in joints))
        done = base.value_as_int(status.msg.motion_status) == 0
        total_error_deg = sum(math.degrees(abs(value)) for value in joints)
        if total_error_deg <= best_total_error_deg - 1.0:
            best_total_error_deg = total_error_deg
            last_progress_at = now
        if error > args.tolerance_deg and now - last_progress_at >= 8.0:
            mode = getattr(status.msg, 'ctrl_mode', 'unavailable')
            motion_mode = getattr(status.msg, 'mode_feedback', 'unavailable')
            states = robot.get_joints_enable_status_list()
            raise RuntimeError(
                f'{args.channel} homing made no meaningful progress for 8 s; '
                f'max_error={error:.2f} deg, total_error={total_error_deg:.2f} deg; '
                f'ctrl_mode={mode}, move_mode={motion_mode}, '
                f'motion_status={status.msg.motion_status}, enabled={states}'
            )
        if now-last_print >= .5:
            angles = ', '.join(f'{math.degrees(value):+.2f}' for value in joints)
            width_text = '' if gripper_width is None else f'; gripper={gripper_width*1000:.1f} mm'
            print(f'{args.channel}: remaining={error:.2f} deg; joints_deg=[{angles}]{width_text}')
            last_print = now
        if (now-started >= .5 and done and error <= args.tolerance_deg
                and (gripper_width is None or gripper_width <= .002)):
            if settled_since is None:
                settled_since = now
            elif now-settled_since >= .5:
                robot.move_j(bounded_hold(joints,config))
                if gripper is not None:
                    gripper.move_gripper_m(0.0, gripper_force)
                detail = 'Gripper closed.' if gripper is not None else 'Gripper unchanged.'
                print(f'{args.channel}: homing complete; enabled and holding. {detail}')
                return
        else:
            settled_since = None
        time.sleep(.02)
    raise TimeoutError(f'Homing did not finish within {args.timeout:g} seconds')


def main(argv=None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    if not args.execute:
        print(f'DRY-RUN: {args.channel}, MOVE_J, {args.speed_percent}% speed, six joints to 0 deg; gripper unchanged.')
        print('Add --execute to connect, enable and home this arm.')
        return 0
    robot = None
    control_attempted = False
    stop = False
    def request_stop(_signum,_frame):
        nonlocal stop
        stop = True
    previous_handlers = {}
    try:
        for signum in (signal.SIGINT,signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum,request_stop)
        robot, config = connect_selected_arm(args)
        base.wait_for_feedback(robot,args.feedback_timeout)
        # Validate before marking the beginning of control.
        wait_complete_joints(
            robot, config, args.feedback_timeout, args.stale_timeout,
            args.joint_limit_tolerance_deg,
        )
        base.ensure_arm_status_normal(robot)
        if stop:
            return 130
        control_attempted = True
        run_home(robot,config,args,lambda:stop)
        return 0
    except (KeyboardInterrupt,InterruptedError):
        if control_attempted:
            robot.electronic_emergency_stop()
            print(f'{args.channel}: interrupted; electronic emergency stop sent.')
        return 130
    except Exception as exc:
        if control_attempted:
            robot.electronic_emergency_stop()
            print(f'{args.channel}: failure; electronic emergency stop sent.')
        else:
            print(f'{args.channel}: preflight refused; no motion or E-stop command sent.')
        print(f'Homing failed: {type(exc).__name__}: {exc}')
        return 1
    finally:
        base.disconnect_quietly(robot)
        for signum,handler in previous_handlers.items():
            signal.signal(signum,handler)


if __name__ == '__main__':
    raise SystemExit(main())
