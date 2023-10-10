from collections import defaultdict, deque
import threading
from io import BytesIO
from queue import Queue
from rlbench.const import colors
import gcsfs
import h5py
import numpy as np
import torch
from ml_collections import ConfigDict
from PIL import Image
from scipy.spatial.transform import Rotation




class RLBenchDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.path = "/home/levi/data/flavaActionDecoderData"

        # config.context_length = 4
        # We want the dataset to start at the context length + 1 so that we can a backward buffer of data
        # of context length
        config.start_index = 0
        config.random_start = False

        config.image_size = 256

        config.state_key = ""
        config.state_dim = 0

        #config.image_key = "front_rgb, left_shoulder_rgb, right_shoulder_rgb, wrist_rgb"
        config.image_key = "front_rgb"

        #config.action_key = "gripper_pose, gripper_open, ignore_collisions"
        config.action_key = "gripper_pose, gripper_open"

        #config.action_dim = 7 + 1 + 1
        config.action_dim = 7 + 1

        config.need_quat_normalize = "gripper_pose, gripper_pose_delta"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    # The init function is going to set all the parameters for the class
    def __init__(
        self,
        update,
        dataset_name="reach_target",
        start_offset_ratio=None,
        split="train",
    ):
        # Update the config with the dict of updates passed into the constructor.
        self.config = self.get_default_config(update)
        assert self.config.path != ""

        # RLBench task name
        self.dataset_name = dataset_name

        # Load the training or validation dataset
        if split == "train":
            path = f"{self.config.path}/{dataset_name}_train.hdf5"
        elif split == "val":
            path = f"{self.config.path}/{dataset_name}_val.hdf5"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Load the dataset from Google Cloud Storage
        if self.config.path.startswith("gs://"):
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(path, cache_type="block"), "r"
            )
        # Load the dataset from local storage
        else:
            self.h5_file = h5py.File(path, "r")

        if self.config.random_start:
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

        #self.tokenizer = self.build_tokenizer()

        # Retrieve the Flava model and processor
        # self.model = FlavaModel.from_pretrained('facebook/flava-full')
        # self.processor = FlavaProcessor.from_pretrained('facebook/flava-full')

        # Create the master index set to access the data samples. List of tuples (str(idx), k) where idx is the string name index of the episode
        # and k is the index of the data sample in the episode where action sequence starts
        self.master_index = []
        n_episodes = len(self.h5_file)
        for idx in range(n_episodes):
            n_obs = len(self.h5_file[str(idx)]['encoder_emb'])
            self.master_index.extend([(str(idx), k) for k in range(n_obs)])

    def __getstate__(self):
        return self.config, self.random_start_offset, self.dataset_name

    def __setstate__(self, state):
        config, random_start_offset, dataset_name = state
        self.__init__(config)
        self.random_start_offset = random_start_offset
        self.dataset_name = dataset_name

    # Required to define this function when subclassing torch.utils.data.Dataset
    def __len__(self):
        num_samples = len(self.master_index)
        return num_samples


    # This function randomizes the starting index of the dataset
    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    # This function is called when the dataset is indexed to get the data sample
    def __getitem__(self, index):
        # Get the conditioned index for the data buffer
        index = self.process_index(index)
        # initialize the dictionary to store the data
        
        
        # Go to the task file and get the instruction for the task
        # with BytesIO(self.h5_file["task_id"][index][0]) as fin:
        #     task_id = fin.read().decode("utf-8")
        # variation_id = self.h5_file["variation_id"][index][0].item()
        # instruct = get_instruct(task_id, variation_id)

        # tokenized_instruct, padding_mask = self.tokenizer(instruct)

        # res["instruct"] = tokenized_instruct
        # res["padding_mask"] = padding_mask
        # res = {"instruct": {}}
        # res["instruct"] = instruct
        
        
        # Concatenate and normalize the action data and add it to the dictionary
        # action_array= np.array([            
        #     np.concatenate(
        #         [
        #             self._normalize_quat(k, self.h5_file[k][index - n])
        #             for k in self.config.action_key.split(", ")
        #         ],
        #         axis=-1,
        #     )            
        #     for n in range(self.config.context_length)
        # ], dtype=np.float32)
        # The first element of the action array is the learning target
        
        
        
        # Loop through the RLBench cameras we define in the config extract the data from the hdf5 file
        # and add them to the dictionary
        # for key in self.config.image_key.split(", "):
        #     img_hold["image"][key] = self.h5_file[key][index]        
        # key = self.config.image_key.split(", ")[0]
       
        # images = [
        #     self.h5_file[key][index - n][0,...]
        #     for n in range(self.config.context_length)
        # ]
        # instructs = [
        #     instruct for n in range(self.config.context_length)
        # ]
        
        # flava_in = self.processor(
        #     text=instructs,
        #     images=images,
        #     return_tensors="pt",
        #     padding="max_length",
        #     max_length=197,
        #     return_codebook_pixels=False,
        #     return_image_mask=False,
        # )

        # flava_out = self.model(**flava_in)
        # multimodal_embeddings = flava_out.multimodal_embeddings

        # Each data sequence is composed of a starting observation and then an sequence of keypoint actions
        # Add the starting observation encoder embedding and the list of keypoint actions
        (ep_idx, obs_idx) = self.master_index[index]

        data = defaultdict(deque)
        data['encoder_emb'].extend([self.h5_file[ep_idx]['encoder_emb'][obs_idx]])
        data['action'].extend(self.h5_file[ep_idx]['kpnt_action'][:])
        data['task_id'].append(self.h5_file[ep_idx]['task_id'][0])
        data['variation_id'].append(self.h5_file[ep_idx]['variation_id'][0])
        # Return the target sequence with eos token
        eos = np.zeros(self.config.action_dim, dtype=np.float32) 
        eos[1::2] = -1 # odd values are -1
        data['action'].append(eos)
        target = data['action'].copy()
        

        # Now add the rest of the encoder embeddings corresponding to the keypoint actions
        data['encoder_emb'].extend(
            [self.h5_file[ep_idx]['encoder_emb'][kpnt_idx] for kpnt_idx in self.h5_file[ep_idx]['kpnt_idx']]
        )

        # Apply a mean pooling to the encoder embeddings to get a single embedding for each
        for emb_idx in range(len(data['encoder_emb'])):
            data['encoder_emb'][emb_idx] = np.mean(data['encoder_emb'][emb_idx], axis=0)

        
        # Add the SOS token to the start of the action sequence (eos already added)
        sos = np.zeros(self.config.action_dim, dtype=np.float32)
        sos[0::2] = -1 # even values are -1
        data['action'].appendleft(sos)

        # Convert the deques to numpy arrays
        data = {k: np.array(v) for k, v in data.items()}
        data = {k: torch.tensor(v) for k, v in data.items()}
        # Convert the target sequence to a torch tensor
        target = torch.tensor(np.array(target))       

        return (data, target)

    def _normalize_quat(self, name, quat):

        if name in self.config.need_quat_normalize.split(", "):
            return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
        else:
            return quat

    @property
    def num_actions(self):
        return self.config.action_dim

    @property
    def obs_shape(self):
        res = {"image": {}}
        for key in self.config.image_key.split(", "):
            res["image"][key] = (256, 256, 3)
        if self.config.state_key != "":
            res["state"] = self.config.state_dim
        return res


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


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def get_cont_action(
    disc_action,
    vox_size=100,
    rotation_resolution=5,
    scene_bound=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
):
    coords = disc_action[:3]
    rot = disc_action[3:-1]
    grip = disc_action[-1]

    bound = np.array(scene_bound)

    res = (bound[3:] - bound[:3]) / vox_size
    trans = bound[:3] + res * coords + res / 2

    cont_action = np.concatenate(
        [trans, discrete_euler_to_quaternion(rot, rotation_resolution), grip]
    )
    # translation [0, vox_size - 1]
    # rotation [0, 360 / vox_size - 1]
    return cont_action


if  __name__ == "__main__":
    from getdata import RLBenchDataset
    import torch
    dataset = RLBenchDataset(
        None,
        dataset_name="reach_target",
        start_offset_ratio=None,
        split="train",
    )

    train_loader = torch.utils.data.DataLoader(    
    dataset=dataset, 
    batch_size=10, 
    shuffle=True, 
    num_workers=0)

    #print the keys to access the data fields for each data sample
    print(f"Training dataset batch info:")
    for i, (data, target) in enumerate(train_loader):
        print("action:", data['action'].shape)
        print("encoder_emb:", data['encoder_emb'].shape)
        print("target:", target.shape)
        break
    print("num batches: " + str(len(train_loader)))
