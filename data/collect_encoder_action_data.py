import os
import random
import time
from collections import defaultdict, deque

import h5py
import numpy as np
import rlbench.backend.task as rlbench_task
from absl import app, flags
from rlbench import ObservationConfig, CameraConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.const import colors

from tqdm.auto import tqdm, trange

from utils import convert_keypoints, keypoint_discovery
from transformers import FlavaProcessor, FlavaModel


FLAGS = flags.FLAGS

flags.DEFINE_string("save_path", "/home/levi/data/flavaActionDecoderData", "Where to save the preprocessed training data")
# flags.DEFINE_list(
#     "tasks",    
#     [
#         "reach_target",
#         "phone_on_base",
#         "pick_and_lift",
#         "pick_up_cup",
#         "put_rubbish_in_bin",
#         "stack_wine",
#         "take_lid_off_saucepan",
#         "take_umbrella_out_of_umbrella_stand",
#     ],
#     "The tasks to collect. If empty, all tasks are collected.",
# )
flags.DEFINE_list(
    "tasks",    
    [
        "reach_target",
    ],
    "The tasks to collect. If empty, all tasks are collected.",
)
flags.DEFINE_list("image_size", [256, 256], "The size of the images to save.")
# An episode is defined as a specific physical starting state (most likely layout/positioning) 
# of the task objects for a particular variation, and these must differ between episodes across 
# the same variation with the same semantic description. At minimum, the positioning/orientation 
# of task objects must vary with episodes
flags.DEFINE_integer(
    "train_episodes_per_task", 100, "The number of episodes to collect per task."
)
flags.DEFINE_integer(
    "val_episodes_per_task", 5, "The number of episodes to collect per task."
)
# A variation is thus defined as a specific semantic meaning of the textual 
# description of the task, such that different variations actually instruct 
# the robot to do something slightly different on every variation. (different colors)
flags.DEFINE_integer(
    "variations", 6, "Number of variations to collect per episode. -1 for all."
)
flags.DEFINE_integer("num_frames", 1, "Number of frames to stack.")
flags.DEFINE_integer("vox_size", 16, "Voxel size to discretize translation.")
flags.DEFINE_integer(
    "rotation_resolution", 5, "Rotation resolution to discretize rotation."
)
flags.DEFINE_list("encoder_emb", [197*2, 768], "The size of the Flava embedding from the encoder, where the first dimension is the text + image token context length and the second is the embedding dim length")

# fmt: off
# ['left_shoulder_depth', 'left_shoulder_mask', 'left_shoulder_point_cloud', 'right_shoulder_depth', 'right_shoulder_mask', 'right_shoulder_point_cloud', 'overhead_depth', 'overhead_rgb', 'overhead_mask', 'overhead_point_cloud', 'wrist_depth', 'wrist_mask', 'wrist_point_cloud', 'front_depth', 'front_mask', 'front_point_cloud', 'joint_forces', 'gripper_pose', 'gripper_matrix', 'gripper_joint_positions', 'gripper_touch_forces', 'task_low_dim_state', 'misc',
# 'front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb', 'joint_positions', 'gripper_open', 'joint_velocities']
# fmt: on

# data fields we want to include or exclude
TO_BE_REMOVED = [
    "misc", 
    "task_low_dim_state",
    "disc_action",
    "gripper_pose_delta",
]
TO_BE_ADDED = [
    "cont_action",
    "time",
    "task_id",
    "variation_id",
    "ignore_collisions",
    "encoder_emb",
]

def get_instruct(task, variation):
    # https://github.com/stepjam/RLBench/tree/master/rlbench/tasks
    if task == "stack_wine":
        return f"place the wine bottle on the wine rack."
    elif task == "take_umbrella_out_of_umbrella_stand":
        return f"grasping the umbrella by its handle, lift it up and out of the stand."
    elif task == "reach_target":
        return f"touch the {colors[variation][0]} ball with the panda gripper."
    elif task == "pick_and_lift":
        return f"pick up the {colors[variation]} block and lift it up to the target."
    elif task == "pick_up_cup":
        return f"grasp the {colors[variation]} cup and lift it."
    elif task == "put_rubbish_in_bin":
        return f"pick up the rubbish and leave it in the trash can."
    elif task == "take_lid_off_saucepan":
        return "grip the saucepan's lid and remove it from the pan."
    elif task == "open_drawer":
        OPTIONS = ["bottom", "middle", "top"]
        return f"grip the {OPTIONS[variation]} handle and pull the {OPTIONS[variation]} drawer open."
    elif task == "meat_off_grill":
        OPTIONS = ["chicken", "steak"]
        return f"pick up the {OPTIONS[variation]} and place it next to the grill."
    elif task == "put_item_in_drawer":
        OPTIONS = ["bottom", "middle", "top"]
        return f"open the {OPTIONS[variation]} drawer and place the block inside of it."
    elif task == "turn_tap":
        OPTIONS = ["left", "right"]
        return "grasp the {OPTIONS[variation]} tap and turn it."
    elif task == "put_groceries_in_cupboard":
        GROCERY_NAMES = [
            "crackers",
            "chocolate jello",
            "strawberry jello",
            "soup",
            "tuna",
            "spam",
            "coffee",
            "mustard",
            "sugar",
        ]
        return f"pick up the {GROCERY_NAMES[variation]} and place it in the cupboard."
    elif task == "place_shape_in_shape_sorter":
        SHAPE_NAMES = ["cube", "cylinder", "triangular prism", "star", "moon"]
        return f"pick up the {SHAPE_NAMES[variation]} and put it in the sorter."
    elif task == "light_bulb_in":
        return f"pick up the light bulb from the {colors[variation]} stand, lift it up to just above the lamp, then screw it down into the lamp in a clockwise fashion."
    elif task == "close_jar":
        return f"close the {colors[variation]} jar."
    elif task == "stack_blocks":
        MAX_STACKED_BLOCKS = 3
        color_index = int(variation / MAX_STACKED_BLOCKS)
        blocks_to_stack = 2 + variation % MAX_STACKED_BLOCKS
        color_name, color_rgb = colors[color_index]
        return (
            f"place {blocks_to_stack} of the {color_name} cubes on top of each other."
        )
    elif task == "put_in_safe":
        safe = {0: "bottom", 1: "middle", 2: "top"}
        return f"put the money away in the safe on the {safe[variation]} shelf."
    # elif task == "push_buttons":
    #     pass
    elif task == "insert_onto_square_peg":
        return f"put the ring on the {colors[variation]} spoke."
    elif task == "stack_cups":
        target_color_name, target_rgb = colors[variation]
        return f"stack the other cups on top of the {target_color_name} cup."
    elif task == "place_cups":  # 3 variations
        return f"place {variation + 1} cups on the cup holder."
    else:
        raise ValueError(f"Unknown task: {task}")

def get_shape_dtype(k, dummy_timestep):
    if k == "gripper_open" or k == "time":
        v_dtype = np.float32
        v_shape = (1,)
    elif k == "cont_action":
        v_dtype = np.float32
        v_shape = (8,)
    elif k == "disc_action":
        v_dtype = np.int32
        v_shape = (7,)
    elif k == "gripper_pose_delta":
        v_dtype = np.float32
        v_shape = (7,)
    elif k == "task_id":
        v_dtype = h5py.special_dtype(vlen=np.dtype("uint8"))
        v_shape = ()
    elif k == "variation_id":
        v_dtype = np.uint8
        v_shape = (1,)
    elif k == "ignore_collisions":
        v_dtype = np.uint8
        v_shape = (1,)
    elif k == "encoder_emb":
        v_dtype = np.float32
        v_shape = (FLAGS.encoder_emb[0], FLAGS.encoder_emb[1])
    else:
        v_dtype = dummy_timestep.__dict__[k].dtype
        v_shape = dummy_timestep.__dict__[k].shape
    return v_shape, v_dtype


# Define the parameters of the hdf5 files we want to create using parameters from the dummy timestep
def create_hdf5(rlbench_env, task, hdf5_name, size=int(1e5)):
    # dummy_task = rlbench_env.get_task(task)
    # dummy_demo = np.array(dummy_task.get_demos(1, live_demos=True)[0])
    # dummy_timestep = dummy_demo[0]

    # keys = list(dummy_timestep.__dict__.keys())
    # # Remove any keys that are None
    # keys = [k for k in keys if dummy_timestep.__dict__[k] is not None]
    # keys.extend(TO_BE_ADDED)
    # # Remove any keys that are in the TO_BE_REMOVED list
    # keys = [k for k in keys if k not in TO_BE_REMOVED]
    

    h5_file = h5py.File(hdf5_name, "x")
    # h5_file_shuffled = h5py.File(hdf5_shuffled_name, "x")

    # for k in keys:
    #     v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)

    #     h5_file.create_dataset(
    #         k,
    #         (size, FLAGS.num_frames, *v_shape),
    #         dtype=v_dtype,
    #         chunks=(16, FLAGS.num_frames, *v_shape),
    #     )
    #     # h5_file_shuffled.create_dataset(
    #     #     k,
    #     #     (size, FLAGS.num_frames, *v_shape),
    #     #     dtype=v_dtype,
    #     #     chunks=(16, FLAGS.num_frames, *v_shape),
    #     # )
    return h5_file


def collect_data(
    rlbench_env,
    task,
    task_name,
    tasks_with_problems,
    h5_file,
    num_episodes,
):
    variation_count = 0
    grp_id = 0

    task_env = rlbench_env.get_task(task)    

    total_data = deque()

    var_target = task_env.variation_count()
    if FLAGS.variations >= 0:
        var_target = np.minimum(FLAGS.variations, var_target)

    print("Task:", task_env.get_name(), "// Variation Target:", var_target)

    # Retrieve the Flava model and processor and put then in the GPU
    flava_model = FlavaModel.from_pretrained('facebook/flava-full')
    flava_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    
    # Iterate through the variations of the task
    while True:
        if variation_count >= var_target:
            break
        task_env.set_variation(variation_count)
        _, _ = task_env.reset()

        abort_variation = False
        # Collect data for each episode of the variation
        for ex_idx in range(num_episodes):
            print(
                "Task:",
                task_env.get_name(),
                "// Variation:",
                variation_count,
                "// Episode:",
                ex_idx,
            )
            attempts = 10
            # Try to collect an episode, if it fails, try again up to 10 times
            while attempts > 0:
                try:
                    (demo,) = task_env.get_demos(amount=1, live_demos=True)
                    print(f"success. demo length {len(demo)}.")
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        "Failed collecting task %s (variation: %d, "
                        "example: %d). Skipping this task/variation.\n%s\n"
                        % (task_env.get_name(), variation_count, ex_idx, str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                
                
                # init the stack where we will assemble the data for each episode
                stack = defaultdict(deque)

                task_instruct = get_instruct(task_env.get_name(), variation_count)

                ### Save keypoint data ###
                # Element indices of the keypoints in the demo
                episode_keypoints = keypoint_discovery(demo)
                # Convert all the keypoints into list of poses at the keypoint indices
                kpnt_cont_action_list, disc_action_list = convert_keypoints(
                    demo, episode_keypoints, action_type="cont",
                )
                # Add the keyframe actions to the stack                
                stack["kpnt_action"].extend(np.array(kpnt_cont_action_list, dtype=np.float32))

                # get the subset of demo observations we want to convert to encoder embeddings
                demo_subset_obs = [demo[i] for i in episode_keypoints]
                kpnt_images = [
                    [obs.left_shoulder_rgb, obs.right_shoulder_rgb, obs.wrist_rgb] for obs in demo_subset_obs
                ]

                # processing the 3 images for each timestep
                for img_set in kpnt_images:
                    # generate a set of instructions to match the set of images going into the encoder                    
                    instruction = [task_instruct for n in range(len(img_set))]                    
                    flava_in = flava_processor(
                        text=instruction,
                        images=img_set,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=52,
                        return_codebook_pixels=False,
                        return_image_mask=False,
                    )
                    flava_out = flava_model(**flava_in)
                    multimodal_embeddings = flava_out.multimodal_output.pooler_output
                    # For each of the multimodal embeddings, add the embeddings to the stack
                    stack["kpnt_encoder_emb"].append(multimodal_embeddings.detach().numpy())

                
                # add the keypoint observation indices to the stack
                stack["kpnt_idx"].extend(np.array(episode_keypoints, dtype=np.int32))
                # add the gripper pose at the keypoint to the stack

                ### Save startpoint data ###
                # Get the subsampled indices of the demo episode for every fifth timestep
                # These will be the indices of the demo where the training sequence will start
                demo_startpoint_indices = np.arange(0, len(demo), 5)
                demo_startpoint_indices = [0]
                # Do not include keypoint indices in the starting point indices 
                demo_startpoint_indices = np.setdiff1d(demo_startpoint_indices, episode_keypoints)           
                # Convert all the startpoint indices into list of poses at the keypoint indices
                startpoint_cont_action_list, disc_action_list = convert_keypoints(
                    demo, demo_startpoint_indices, action_type="cont",
                )
                # Add the keyframe actions to the stack                
                stack["startpoint_action"].extend(np.array(startpoint_cont_action_list, dtype=np.float32))
                # get the subset of demo observations we want to convert to encoder embeddings
                demo_subset_obs = [demo[i] for i in demo_startpoint_indices]
                
                startpoint_images = [
                    [obs.left_shoulder_rgb, obs.right_shoulder_rgb, obs.wrist_rgb] for obs in demo_subset_obs
                ]
                
                # processing the 3 images for each timestep
                for img_set in startpoint_images:
                    # generate a set of instructions to match the set of images going into the encoder                    
                    instruction = [task_instruct for n in range(len(img_set))]                    
                    flava_in = flava_processor(
                        text=instruction,
                        images=img_set,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=52,
                        return_codebook_pixels=False,
                        return_image_mask=False,
                    )
                    flava_out = flava_model(**flava_in)
                    multimodal_embeddings = flava_out.multimodal_output.pooler_output
                    # For each of the multimodal embeddings, add the embeddings to the stack
                    stack["startpoint_encoder_emb"].append(multimodal_embeddings.detach().numpy())
                
                # Add the encoder_indices to the stack
                stack["startpoint_idx"].extend(np.array(demo_startpoint_indices, dtype=np.int32))

                # Add the episode and variation id to the stack
                stack["task_id"].append(np.array(ex_idx, dtype=np.uint8))
                stack["variation_id"].append(np.array(variation_count, dtype=np.uint8))
                #stack["instruction"] = task_instruct

                # Each element of this list contains all the data for each episode and task variation
                total_data.append(stack)
                
               
                break
            if abort_variation:
                break

        variation_count += 1

    start = time.process_time()

    # Write the total data struct to the h5 file creating groups and datasets for each key
    for i, epi_dat in enumerate(total_data):
        h5_grp = h5_file.create_group(f"{i:04d}")
        for k in epi_dat.keys():
            # create the data set for each key
            h5_dset = h5_grp.create_dataset(k, data=epi_dat[k])
            # assign the data to the h5 dataset
            h5_dset[:] = epi_dat[k]

    h5_file.close()


    print(time.process_time() - start)

    



def create_env():
    img_size = list(map(int, FLAGS.image_size))

    CAMERA_CONFIG_ON = CameraConfig(
        rgb=True, 
        image_size=img_size, 
        depth=False, 
        mask=False, 
        point_cloud=False)

    CAMERA_CONFIG_OFF = CameraConfig(
        rgb=False, 
        image_size=img_size, 
        depth=False, 
        mask=False, 
        point_cloud=False)


    obs_config = ObservationConfig(
        left_shoulder_camera=CAMERA_CONFIG_ON,
        right_shoulder_camera=CAMERA_CONFIG_ON,
        overhead_camera=CAMERA_CONFIG_OFF,
        wrist_camera=CAMERA_CONFIG_ON,
        front_camera=CAMERA_CONFIG_OFF

    )
    obs_config.set_all_low_dim(True)
    

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
    )
    rlbench_env = Environment(action_mode, obs_config=obs_config, headless=True)
    rlbench_env.launch()

    return rlbench_env


def main(argv):
    # Get the list of task files from the built in tasks folder
    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]
    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError(f"Task {t} not recognised!.")
        task_files = FLAGS.tasks

    # Convert the task files to task classes and create the RLBench environment
    tasks = [task_file_to_task_class(t) for t in task_files]
    num_tasks = len(tasks)
    tasks_with_problems = ""

    rlbench_env = create_env()

    # For each task we want to collect data for
    for task_index in range(num_tasks):
        task_name = FLAGS.tasks[task_index]

        # train
        train_hdf5_name = os.path.join(FLAGS.save_path, f"{task_name}_train.hdf5")
        # train_hdf5_shuffled_name = os.path.join(
        #     FLAGS.save_path, f"{task_name}_train_shuffled.hdf5"
        # )
        try:
            os.remove(train_hdf5_name)
        except OSError:
            pass

        h5_file = create_hdf5(
            rlbench_env,
            tasks[task_index],
            train_hdf5_name,
        )
        collect_data(
            rlbench_env,
            tasks[task_index],
            task_name,
            tasks_with_problems,
            h5_file,
            num_episodes=FLAGS.train_episodes_per_task,
        )

        # val
        val_hdf5_name = os.path.join(FLAGS.save_path, f"{task_name}_val.hdf5")
        # val_hdf5_shuffled_name = os.path.join(
        #     FLAGS.save_path, f"{task_name}_val_shuffled.hdf5"
        # )
        try:
            os.remove(val_hdf5_name)
        except OSError:
            pass

        h5_file = create_hdf5(
            rlbench_env,
            tasks[task_index],
            val_hdf5_name,
        )
        collect_data(
            rlbench_env,
            tasks[task_index],
            task_name,
            tasks_with_problems,
            h5_file,
            num_episodes=FLAGS.val_episodes_per_task,
        )

    print(tasks_with_problems)

    # combine_multi_task(rlbench_env, tasks, num_tasks)

    rlbench_env.shutdown()

    # test(argv)


def combine_multi_task(rlbench_env, tasks, num_tasks):

    dummy_task = rlbench_env.get_task(tasks[0])
    dummy_demo = np.array(dummy_task.get_demos(1, live_demos=True)[0])
    dummy_timestep = dummy_demo[0]
    keys = list(dummy_timestep.__dict__.keys())
    keys = [k for k in keys if dummy_timestep.__dict__[k] is not None]
    keys.extend(TO_BE_ADDED)
    # Remove any keys that are in the TO_BE_REMOVED list
    keys = [k for k in keys if k not in TO_BE_REMOVED]

    combine_hdf5_files(
        rlbench_env, tasks, num_tasks, dummy_timestep, keys, split="train"
    )

    combine_hdf5_files(rlbench_env, tasks, num_tasks, dummy_timestep, keys, split="val")


def combine_hdf5_files(
    rlbench_env, tasks, num_tasks, dummy_timestep, keys, split="train"
):
    print("combining hdf5 files for", split)

    total_data_fn = os.path.join(FLAGS.save_path, f"multi_task_{split}.hdf5")
    total_data_shuffled_fn = os.path.join(
        FLAGS.save_path, f"multi_task_{split}_shuffled.hdf5"
    )

    try:
        os.remove(total_data_fn)
        os.remove(total_data_shuffled_fn)
    except OSError:
        pass

    total_data, total_data_shuffled = create_hdf5(
        rlbench_env,
        tasks[0],
        total_data_fn,
        total_data_shuffled_fn,
        size=int(1e5) * num_tasks,
    )

    total_timestep = 0
    for task_index in trange(num_tasks, desc="combining data", ncols=0):
        task_name = FLAGS.tasks[task_index]

        h5_file_name = os.path.join(
            FLAGS.save_path, f"{task_name}_{split}_shuffled.hdf5"
        )
        h5_file = h5py.File(h5_file_name, "r")
        h5_size = h5_file[keys[0]].shape[0]

        for k in keys:
            total_data[k][total_timestep : total_timestep + h5_size] = h5_file[k][:]

        total_timestep += h5_file[k].shape[0]

    for k in keys:

        v_shape, v_dtype = get_shape_dtype(k, dummy_timestep)

        total_data[k].resize((total_timestep, FLAGS.num_frames, *v_shape))
        total_data_shuffled[k].resize((total_timestep, FLAGS.num_frames, *v_shape))

    indices = list(range(total_timestep))
    random.shuffle(indices)

    for i, j in enumerate(tqdm(indices, desc="shuffling", ncols=0)):
        for k in keys:
            if k == "task_id":
                total_data_shuffled[k][i] = total_data[k][j][0]
            else:
                total_data_shuffled[k][i] = total_data[k][j]


def test(argv):
    h5_file = h5py.File(
        os.path.join(FLAGS.save_path, "multi_task_train_shuffled.hdf5"), "r"
    )
    for k in [
        "joint_velocities",
        "joint_positions",
        "gripper_pose",
        "gripper_joint_positions",
        "gripper_pose_delta",
    ]:
        print(k, h5_file[k].shape)
    for k in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]:
        print(k, h5_file[k].shape)


if __name__ == "__main__":
    app.run(main)  # run and test
    # app.run(test) # test only
