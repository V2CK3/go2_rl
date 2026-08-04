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
        episode_length_s = 20

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
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.15, 0.15, 0.7, 0.0, 0.0]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 5.
        heading_command = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.42]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
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
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_calculated_torque = False
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_joint_friction = False
        joint_friction_range = [0.01, 1.15]

        randomize_joint_damping = False
        joint_damping_range = [0.3, 1.5]

        randomize_joint_armature = False
        joint_armature_range = [0.0001, 0.05]

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
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            base_height = -1.0
            torques = -0.00005
            dof_acc = -1.5e-7
            dof_vel = 0.0
            collision = -1.
            action_rate = -0.01
            feet_air_time = 1.0
            default_pos = -0.01

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.3
        max_contact_force = 100.
        cycle_time = 0.5
        target_foot_height = 0.2

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
            dof_vel = 1.5
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
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-5
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        sym_loss = False
        obs_permutation = [
            0.0001, -1, -2,
            -3, 4, -5, -6, 7, -8,
            -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,
            -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,
            -36, 37, 38, -33, 34, 35, -42, 43, 44, -39, 40, 41,
        ]
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
        frame_stack = 1
        sym_coef = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 30000
        save_interval = 100
        experiment_name = 'go2_stairs'
        run_name = 'stairs'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
