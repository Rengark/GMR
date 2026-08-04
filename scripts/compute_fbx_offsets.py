"""Compute rotation offsets for fbx_to_g1.json by comparing FBX T-pose quaternions
with MuJoCo robot T-pose body orientations.

Formula: R_offset = Q_fbx_tpose.inv() * Q_robot_tpose
"""
import argparse
import json
import pickle
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from pathlib import Path

ASSET_ROOT = Path(__file__).parent.parent / "assets"
IK_CONFIG_ROOT = Path(__file__).parent.parent / "general_motion_retargeting" / "ik_configs"

JOINT_MAP = {
    "pelvis": "Hips",
    "left_hip_roll_link": "LeftUpLeg",
    "left_knee_link": "LeftLeg",
    "left_toe_link": "LeftToeBase",
    "right_hip_roll_link": "RightUpLeg",
    "right_knee_link": "RightLeg",
    "right_toe_link": "RightToeBase",
    "torso_link": "Spine1",
    "left_shoulder_yaw_link": "LeftArm",
    "left_elbow_link": "LeftForeArm",
    "left_wrist_yaw_link": "LeftHand",
    "right_shoulder_yaw_link": "RightArm",
    "right_elbow_link": "RightForeArm",
    "right_wrist_yaw_link": "RightHand",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True, help="FBX motion pickle (frame 0 = T-pose)")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--write_config", action="store_true", help="Write computed offsets to fbx_to_g1.json")
    args = parser.parse_args()

    # Load FBX T-pose from pickle (frame 0)
    with open(args.motion_file, "rb") as f:
        motion_data = pickle.load(f)

    tpose_frame = motion_data[0]  # dict: joint_name -> [pos_list, quat_wxyz_list]

    # Load robot model and get T-pose body quaternions
    xml_path = str(ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml")
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    mj.mj_forward(model, data)  # qpos=0 is T-pose

    print("=" * 60)
    print("FBX T-pose quaternions (wxyz) vs Robot T-pose quaternions (wxyz)")
    print("=" * 60)

    computed_offsets = {}

    for robot_body, human_joint in JOINT_MAP.items():
        # Get FBX T-pose quaternion (stored as wxyz in pickle)
        if human_joint not in tpose_frame:
            print(f"WARNING: {human_joint} not found in FBX data, skipping {robot_body}")
            continue

        fbx_quat_wxyz = np.array(tpose_frame[human_joint][1])
        R_fbx = R.from_quat(fbx_quat_wxyz, scalar_first=True)

        # Get robot T-pose quaternion from MuJoCo (xquat is wxyz)
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot_body)
        robot_quat_wxyz = data.xquat[body_id].copy()
        R_robot = R.from_quat(robot_quat_wxyz, scalar_first=True)

        # Compute offset: R_offset = R_fbx.inv() * R_robot
        R_offset = R_fbx.inv() * R_robot
        offset_wxyz = R_offset.as_quat(scalar_first=True)

        # Round small values to zero for cleanliness
        offset_wxyz = np.round(offset_wxyz, 8)
        offset_wxyz[np.abs(offset_wxyz) < 1e-7] = 0.0

        computed_offsets[robot_body] = offset_wxyz.tolist()

        print(f"\n{robot_body} <- {human_joint}:")
        print(f"  FBX T-pose (wxyz):   {fbx_quat_wxyz}")
        print(f"  Robot T-pose (wxyz): {robot_quat_wxyz}")
        print(f"  R_offset (wxyz):     {offset_wxyz}")

        # Verify: R_fbx * R_offset should equal R_robot
        R_check = R_fbx * R_offset
        check_quat = R_check.as_quat(scalar_first=True)
        error = np.linalg.norm(check_quat - robot_quat_wxyz)
        # Handle quaternion double-cover (q and -q are same rotation)
        error = min(error, np.linalg.norm(check_quat + robot_quat_wxyz))
        print(f"  Verification error:  {error:.8f}")

    print("\n" + "=" * 60)
    print("Computed offsets (paste into fbx_to_g1.json):")
    print("=" * 60)
    for robot_body, offset in computed_offsets.items():
        print(f'  "{robot_body}": {offset}')

    if args.write_config:
        config_path = IK_CONFIG_ROOT / "fbx_to_g1.json"
        with open(config_path) as f:
            config = json.load(f)

        for table_key in ["ik_match_table1", "ik_match_table2"]:
            for robot_body, offset in computed_offsets.items():
                if robot_body in config[table_key]:
                    config[table_key][robot_body][4] = offset

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"\nWritten to {config_path}")


if __name__ == "__main__":
    main()
