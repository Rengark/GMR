"""Fix arm rotation offsets.

The robot at qpos=0 has arms hanging DOWN (identity orientation).
But the FBX T-pose has arms OUT to the sides.
The correct IK target for arms-out is NOT identity - it's Rz(+90) for left, Rz(-90) for right.

Formula: R_offset = R_fbx_tpose.inv() * R_desired_target
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import json
from pathlib import Path

IK_CONFIG_PATH = Path(__file__).parent.parent / "general_motion_retargeting" / "ik_configs" / "fbx_to_g1.json"

with open("/home/sctd/anim/anim/anim_test.pkl", "rb") as f:
    motion = pickle.load(f)
frame0 = motion[0]

# Desired IK target orientations when arms are extended to sides:
# Left arm along +Y: body x-axis should point +Y -> rotate +90 about Z
# Right arm along -Y: body x-axis should point -Y -> rotate -90 about Z
R_left_target = R.from_euler('zy', [90, 90], degrees=True)
R_right_target = R.from_euler('z', 0, degrees=True)

arm_joints = {
    'left_shoulder_yaw_link': ('LeftArm', R_left_target),
    'left_elbow_link': ('LeftForeArm', R_left_target),
    'left_wrist_yaw_link': ('LeftHand', R_left_target),
    'right_shoulder_yaw_link': ('RightArm', R_right_target),
    'right_elbow_link': ('RightForeArm', R_right_target),
    'right_wrist_yaw_link': ('RightHand', R_right_target),
}

print("Corrected arm offsets (wxyz):")
arm_offsets = {}
for robot_body, (human_joint, R_target) in arm_joints.items():
    fbx_quat = np.array(frame0[human_joint][1])
    R_fbx = R.from_quat(fbx_quat, scalar_first=True)
    R_offset = R_fbx.inv() * R_target
    offset_wxyz = R_offset.as_quat(scalar_first=True)
    offset_wxyz = np.round(offset_wxyz, 8)
    offset_wxyz[np.abs(offset_wxyz) < 1e-7] = 0.0

    # Verify: R_fbx * R_offset should give the target
    R_check = R_fbx * R_offset
    check_dir = R_check.apply([1, 0, 0])

    arm_offsets[robot_body] = offset_wxyz.tolist()
    print(f"  {robot_body}: {offset_wxyz.tolist()}")
    print(f"    verify x-axis points: {np.round(check_dir, 4)}")

# Write to config
with open(IK_CONFIG_PATH) as f:
    config = json.load(f)

for table_key in ["ik_match_table1", "ik_match_table2"]:
    for robot_body, offset in arm_offsets.items():
        if robot_body in config[table_key]:
            config[table_key][robot_body][4] = offset

with open(IK_CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=4)

print(f"\nWritten to {IK_CONFIG_PATH}")
