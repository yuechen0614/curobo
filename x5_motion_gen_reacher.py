# ============================================================
# CRITICAL: import warp and curobo BEFORE SimulationApp.
# SimulationApp loads the omni.warp.core extension which
# prepends an older warp (1.8.2) to sys.path.  By importing
# warp and curobo first, the correct version (1.12+) stays
# cached in sys.modules and is never replaced.
# ============================================================
import warp  # locks warp 1.12.x into sys.modules

import torch

a = torch.zeros(4, device="cuda:0")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

# ============================================================
# Only after curobo is loaded, start Isaac Sim
# ============================================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket]",
)
parser.add_argument(
    "--robot",
    type=str,
    default="/home/kaslensu/workspace/isaacsim_project/curobo/x5.yml",
    help="Absolute path to curobo v2 robot config (x5.yml)",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    default=False,
    help="Visualize robot collision spheres",
)
args = parser.parse_args()

try:
    import isaacsim
except ImportError:
    pass

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# ============================================================
# Post-SimulationApp imports (USD / Isaac Sim API available)
# ============================================================
import numpy as np
import carb
import yaml
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.types import ArticulationAction

from curobo._src.util.usd_writer import set_prim_transform

# ---------------------------------------------------------------------------
# Isaac Sim helpers (inlined — no dependency on old curobo v1 helper.py)
# ---------------------------------------------------------------------------

ISAAC_SIM_45 = False
try:
    from omni.importer.urdf import _urdf
except ImportError:
    from isaacsim.asset.importer.urdf import _urdf
    ISAAC_SIM_45 = True


def add_extensions(sim_app, headless_mode=None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        ext_list += ["omni.kit.livestream." + headless_mode]
    for ext in ext_list:
        enable_extension(ext)
    sim_app.update()


def add_robot_to_scene(robot_config, my_world, position=np.array([0, 0, 0])):
    """Load robot URDF into Isaac Sim and return (Robot, prim_path)."""
    import os
    import omni.kit.commands
    import omni.usd

    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = True
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = True
    import_config.default_drive_strength = 100.0
    import_config.default_position_drive_damping = 40.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    urdf_full_path = robot_config["kinematics"]["urdf_path"]
    robot_dir = os.path.dirname(urdf_full_path)
    urdf_filename = os.path.basename(urdf_full_path)
    base_link_name = robot_config["kinematics"]["base_link"]

    if ISAAC_SIM_45:
        dest_path = os.path.join(robot_dir, urdf_filename.replace(".urdf", "_temp.usd"))
        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_full_path,
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:
        imported_robot = urdf_interface.parse_urdf(robot_dir, urdf_filename, import_config)
        robot_path = urdf_interface.import_robot(
            robot_dir, urdf_filename, imported_robot, import_config, ""
        )

    robot_p = Robot(prim_path=robot_path + "/" + base_link_name, name="robot")
    linkp = robot_p.prim.GetStage().GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if ISAAC_SIM_45:
        my_world.initialize_physics()
        robot.initialize()

    return robot, robot_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # Red cube — drag to set IK goal
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0.0, 0.5]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        color=np.array([1.0, 0.0, 0.0]),
        size=0.05,
    )

    # Visual table matching collision_table.yml (4x4x0.2, center z=-0.1)
    cuboid.VisualCuboid(
        "/World/table",
        position=np.array([0.0, 0.0, -0.1]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        color=np.array([0.5, 0.4, 0.3]),
        size=1.0,
        scale=np.array([4.0, 4.0, 0.2]),
    )

    # Load robot config (curobo v2 flat kinematics yaml)
    with open(args.robot) as f:
        robot_cfg = yaml.safe_load(f)

    cspace = robot_cfg["kinematics"]["cspace"]
    j_names = cspace["joint_names"]
    default_config = cspace["default_joint_position"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    # CuRobo v2 motion planner with builtin collision_table scene
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model="collision_table.yml",
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
    )
    planner = MotionPlanner(planner_cfg)
    print("Warming up CuRobo...")
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    print("CuRobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    my_world.scene.add_default_ground_plane()

    articulation_controller = None
    cmd_plan = None
    cmd_idx = 0
    best_seed = 0
    idx_list = []
    n_steps = 0
    i = 0
    past_pose = None
    past_orientation = None
    target_pose = None
    target_orientation = None
    spheres = None

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index

        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()

        if step_index < 10:
            robot._articulation_view.initialize()
            dof_idx_list = [robot.get_dof_index(x) for x in j_names]
            all_zero = np.zeros(len(robot.dof_names))
            robot.set_joint_positions(all_zero, list(range(len(robot.dof_names))))
            robot.set_joint_positions(default_config, dof_idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000.0] * len(dof_idx_list)),
                joint_indices=dof_idx_list,
            )
        if step_index < 20:
            continue

        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position.copy()
        if target_pose is None:
            target_pose = cube_position.copy()
        if target_orientation is None:
            target_orientation = cube_orientation.copy()
        if past_orientation is None:
            past_orientation = cube_orientation.copy()

        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        if np.any(np.isnan(sim_js.positions)):
            carb.log_error("Isaac Sim returned NaN joint positions.")
            continue

        sim_js_names = robot.dof_names

        full_js = JointState.from_position(
            torch.tensor(sim_js.positions, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=list(sim_js_names),
        )
        cu_js = planner.kinematics.get_active_js(full_js)
        cu_js.velocity = torch.zeros_like(cu_js.position)
        cu_js.acceleration = torch.zeros_like(cu_js.position)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = planner.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.array(s.pose[:3]),
                        radius=float(s.radius),
                        color=np.array([0.0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.pose[0]):
                        spheres[si].set_world_pose(position=np.array(s.pose[:3]))
                        spheres[si].set_radius(float(s.radius))

        robot_static = np.nanmax(np.abs(sim_js.velocities)) < 2.0

        cube_moved_from_target = (
            np.linalg.norm(cube_position - target_pose) > 1e-3
            or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
        )
        cube_stationary = (
            np.linalg.norm(past_pose - cube_position) < 1e-4
            and np.linalg.norm(past_orientation - cube_orientation) < 1e-4
        )

        if step_index % 100 == 0:
            print(
                f"[step {step_index}] static={robot_static} "
                f"cube_moved={cube_moved_from_target} cube_stationary={cube_stationary}"
            )

        if cube_moved_from_target and cube_stationary and robot_static:
            goal_pose = GoalToolPose(
                tool_frames=planner.tool_frames,
                position=torch.tensor(
                    [[[[[float(cube_position[0]), float(cube_position[1]), float(cube_position[2])]]]]],
                    device="cuda",
                    dtype=torch.float32,
                ),
                quaternion=torch.tensor(
                    [[[[[ float(cube_orientation[0]), float(cube_orientation[1]),
                          float(cube_orientation[2]), float(cube_orientation[3]) ]]]]],
                    device="cuda",
                    dtype=torch.float32,
                ),
            )

            result = planner.plan_pose(goal_pose, cu_js)

            if result is not None and result.success.any():
                print(f"Plan succeeded (total_time={result.total_time:.3f}s)")
                cmd_plan = result.get_interpolated_plan()
                # cmd_plan.position has shape (batch, num_seeds, T, n_dof); pick best seed
                best_seed = int(result.seed_rank[0, 0]) if result.seed_rank is not None else 0
                n_steps = cmd_plan.position.shape[-2]

                plan_jnames = cmd_plan.joint_names
                idx_list = [robot.get_dof_index(x) for x in plan_jnames
                            if robot.get_dof_index(x) != -1]
                valid_jnames = [x for x in plan_jnames if robot.get_dof_index(x) != -1]
                if valid_jnames != list(plan_jnames):
                    cmd_plan = cmd_plan.reorder(valid_jnames)
                    n_steps = cmd_plan.position.shape[-2]

                cmd_idx = 0
            else:
                status = getattr(result, "status", "unknown") if result is not None else "no result"
                carb.log_warn(f"Plan failed: {status}")

            target_pose = cube_position.copy()
            target_orientation = cube_orientation.copy()

        past_pose = cube_position.copy()
        past_orientation = cube_orientation.copy()

        if cmd_plan is not None and cmd_idx < n_steps:
            pos = cmd_plan.position[0, best_seed, cmd_idx, :].cpu().numpy()
            vel_tensor = cmd_plan.velocity
            vel = vel_tensor[0, best_seed, cmd_idx, :].cpu().numpy() if vel_tensor is not None else None
            art_action = ArticulationAction(
                joint_positions=pos,
                joint_velocities=vel,
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= n_steps:
                cmd_idx = 0
                cmd_plan = None

    simulation_app.close()


if __name__ == "__main__":
    main()
