"""Restore one Piper controller to follower (motion-output) role, without joint targets."""
from __future__ import annotations

import argparse

from ..core.robot import piper as base


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        '--channel', required=True,
        help='one CAN channel, e.g. can_leader1',
    )
    parser.add_argument('--interface', default='socketcan')
    parser.add_argument('--firmware', choices=('default', 'v183', 'v188', 'v189'), default='v189')
    parser.add_argument('--confirm-drop', action='store_true',
                        help='confirm the arm is physically supported if role change releases holding torque')
    base.add_execute_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if not args.channel.strip():
        raise ValueError('channel must not be empty')
    if not args.execute:
        print(f'DRY-RUN: would send follower-role 0x470 FC to {args.channel}; no joint or enable command.')
        print('Support the arm, then add --confirm-drop --execute; power-cycle afterward.')
        return 0
    if not args.confirm_drop:
        raise RuntimeError('Support the arm and add --confirm-drop before changing its role')

    _, factory, _, _ = base._load_sdk()
    config = base.build_config(args)
    robot = factory.create_arm(config)
    try:
        # A leader may not publish normal joint feedback. Connect directly;
        # do not wait for 0x2A5/0x2A6/0x2A7 or send any joint target.
        robot.connect()
        robot.set_follower_mode()
        print(f'{args.channel}: follower-role command sent; no joint motion or enable command sent.')
        print('Keep the arm supported and power-cycle its controller for the role change to take effect.')
        return 0
    finally:
        base.disconnect_quietly(robot)


if __name__ == '__main__':
    raise SystemExit(main())
