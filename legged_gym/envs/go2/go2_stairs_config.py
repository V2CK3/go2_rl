from legged_gym.envs.base.base_config import BaseConfig


class Go2StairsCfg(BaseConfig):
    """Go2 stairs locomotion config (ported from My_unitree_go2_gym GO2_Stairs)."""

    class env:
        short_frame_stack = 1
        frame_stack = 1
        c_frame_stack = 1
        num_envs = 4096
        num_single_obs = 45
        num_observations = int(frame_stack * num_single_obs)
        # actor(45) + base_lin_vel(3) + measured_heights(187)
        single_num_privileged_obs = 45 + 3 + 187
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 16

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        # max_init=1 + demote pinned everyone at L0 in 18-58-23; start at 2 for L1–L2 practice.
        max_init_terrain_level = 2
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # [smooth slope, rough slope, stairs up, stairs down, discrete]
        # Bias stairs-up; keep some down for descent robustness.
        terrain_proportions = [0.05, 0.05, 0.55, 0.25, 0.1]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 5.
        heading_command = False
        # Base env zeros cmds with ||v_xy||<0.2 — keep samples above this.
        deadzone = 0.2

        class ranges:
            # Stay above deadzone so standing cannot farm tracking on cmd=0.
            lin_vel_x = [0.30, 0.55]
            lin_vel_y = [-0.08, 0.08]
            # Keep small; most episodes force yaw=0 in resample (match play/sim2sim).
            ang_vel_yaw = [-0.20, 0.20]
            heading = [-3.14, 3.14]
        # Fraction of resamples that force straight cmd (vy=0, yaw=0).
        straight_command_prob = 0.7

    class init_state:
        # Higher spawn reduces nose-plant on drop-in (prior z=0.42 → front collapse).
        pos = [0.0, 0.0, 0.45]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        # Keep resets on the flat center platform (~3×3 m); leave margin before risers.
        xy_spawn_noise = 1.2
        default_joint_angles = {
            # Symmetric standing pose: front/rear asymmetry caused nose-down + big rear swing.
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.75,
            'RL_thigh_joint': 0.75,
            'FR_thigh_joint': 0.75,
            'RR_thigh_joint': 0.75,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class control:
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        knee_name = "calf"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.25]

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        continuous_push = False
        max_push_force = 0.5
        max_push_torque = 0.5
        push_force_noise = 0.5
        push_torque_noise = 0.5

        randomize_base_mass = True
        added_base_mass_range = [-1, 1]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_pd_gains = True
        # Wider PD rand: MuJoCo/sim2sim effective tracking is softer than nominal Isaac.
        stiffness_multiplier_range = [0.8, 1.15]
        damping_multiplier_range = [0.8, 1.2]

        randomize_calculated_torque = False
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.04, 0.04]

        # Milder joint lag: damping∈[0,2] taught bang-bang calf chatter vs sticky DOFs.
        randomize_joint_friction = True
        joint_friction_range = [0.0, 0.12]

        randomize_joint_damping = True
        joint_damping_range = [0.0, 1.0]

        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.015]

        add_obs_latency = False
        randomize_obs_motor_latency = False
        range_obs_motor_latency = [1, 3]
        randomize_obs_imu_latency = False
        range_obs_imu_latency = [1, 3]

        add_cmd_action_latency = False
        randomize_cmd_action_latency = False
        range_cmd_action_latency = [1, 3]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.30
            orientation = -2.0
            pitch_forward = -2.0
            nose_plant = -1.0
            base_height = -2.0
            torques = -0.00005
            # 18-58-23: stacked jerk/acc penalties zeroed only_positive reward (smoothness→-60).
            dof_acc = -2.0e-7
            dof_vel = -1.0e-4
            calf_acc = -4.0e-7
            collision = -1.2
            action_rate = -0.04
            action_smoothness = -0.01
            lin_vel_smooth = -0.008
            feet_air_time = 4.5
            feet_clearance = 1.6
            feet_clearance_terrain = -1.2
            feet_stumble = -2.5
            feet_contact_forces = -0.01
            foot_slip = -0.10
            drag_gait = -0.4
            commanded_still = -1.5
            uncommanded_yaw = -1.2
            uncommanded_vy = -1.0
            default_hip_pos = -0.6
            default_pos = -0.08
            thigh_overflex = -0.6
            front_rear_thigh_amp = -0.4
            dof_pos_limits = -1.0

        only_positive_rewards = True
        drag_tracking_scale = 0.5
        # 0.14 made tracking too sparse vs penalties; 0.20 still tighter than old 0.25.
        tracking_sigma = 0.20
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.35
        max_contact_force = 120.
        cycle_time = 0.5
        target_foot_height = 0.14
        min_feet_air_time = 0.08
        clearance_terrain_margin = 0.05
        clearance_look_ahead = 0.10
        commanded_still_speed = 0.20
        uncommanded_yaw_cmd_thr = 0.1
        uncommanded_vy_cmd_thr = 0.05
        thigh_overflex_threshold = 1.30
        base_height_clip = 0.25
        lin_vel_z_clip = 2.0
        action_rate_clip = 16.0
        action_smoothness_clip = 8.0
        lin_vel_smooth_clip = 40.0
        # 0.28+demote pinned terrain_level≈0.08; ease promote, demote only by distance.
        terrain_track_up_threshold = 0.18
        terrain_ang_track_up_threshold = 0.12
        terrain_promote_distance_frac = 0.25
        terrain_demote_distance_frac = 0.10
        terrain_demote_on_poor_track = False

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            # High dof_vel noise → reactive PD chatter on calves.
            dof_vel = 0.8
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            quat = 0.1
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class Go2StairsCfgPPO(BaseConfig):
    """PPO config for Go2 stairs."""

    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        # Lower exploration noise — std=1 kept high-freq action chatter late in training.
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        # Prior run noise_std exploded 1→7; lower entropy to keep exploration bounded.
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-5
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_learning_rate = 3.e-4  # prevent adaptive LR from sticking at 1e-2 after collapse
        max_grad_norm = 1.
        # Enforce L/R symmetry to stop single-leg splay / unloaded RL stance.
        sym_loss = True
        obs_permutation = [
            0.0001, -1, -2,
            -3, 4, -5, -6, 7, -8,
            -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,
            -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,
            -36, 37, 38, -33, 34, 35, -42, 43, 44, -39, 40, 41,
        ]
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
        frame_stack = 1
        # Stronger L/R symmetry → less one-sided right-yaw drift in play/sim2sim.
        sym_coef = 2.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 200
        plot_interval = 200
        experiment_name = 'go2_stairs'
        run_name = 'stairs'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
