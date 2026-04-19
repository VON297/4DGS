import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.cam_folder = cam_folder
        self._missing_confidence_warned_cameras = set()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
        
    
    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        image_length = len([f for f in os.listdir(os.path.join(cam_folder,"cam01")) if f.lower().endswith((".jpg", ".png"))])
        #len_cam=len(cam_extrinsics)
        image_paths=[]
        image_poses=[]
        image_times=[]
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            number = os.path.basename(extr.name)[5:-4]
            images_folder=os.path.join(cam_folder,"cam"+number.zfill(2))

            image_range=range(image_length)
            if split=="test":
                image_range = [image_range[0],image_range[int(image_length/3)],image_range[int(image_length*2/3)]]

            for i in image_range:    
                num=i+1
                frame_name = "frame_"+str(num).zfill(5)
                image_path_png = os.path.join(images_folder, frame_name+".png")
                image_path_jpg = os.path.join(images_folder, frame_name+".jpg")
                if os.path.exists(image_path_png):
                    image_path = image_path_png
                elif os.path.exists(image_path_jpg):
                    image_path = image_path_jpg
                else:
                    raise FileNotFoundError(f"Missing image for {frame_name} in {images_folder}")
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i/image_length))

        return image_paths, image_poses,image_times
    

    def _parse_cam_name_from_path(self, image_path):
        return os.path.basename(os.path.dirname(image_path))

    def _is_real_view(self, cam_name):
        try:
            cam_idx = int(cam_name.replace("cam", ""))
        except ValueError:
            return False
        return 1 <= cam_idx <= 4

    def _confidence_path_from_image_path(self, image_path):
        scene_dir = os.path.dirname(os.path.dirname(image_path))
        cam_name = self._parse_cam_name_from_path(image_path)
        frame_name = os.path.basename(image_path)
        return os.path.join(scene_dir, "confidence", cam_name, os.path.splitext(frame_name)[0] + ".png")

    def _load_image_alpha_confidence(self, image_path):
        img_pil = Image.open(image_path)
        if img_pil.mode == "RGBA":
            img_tensor = self.transform(img_pil)
            rgb_tensor = img_tensor[:3, :, :]
            alpha_mask = img_tensor[3:4, :, :]
        else:
            rgb_img = img_pil.convert("RGB")
            rgb_tensor = self.transform(rgb_img)
            alpha_mask = torch.ones((1, rgb_tensor.shape[1], rgb_tensor.shape[2]), dtype=rgb_tensor.dtype)

        cam_name = self._parse_cam_name_from_path(image_path)
        is_real_view = self._is_real_view(cam_name)

        confidence = torch.ones((1, rgb_tensor.shape[1], rgb_tensor.shape[2]), dtype=rgb_tensor.dtype)
        if not is_real_view:
            confidence_path = self._confidence_path_from_image_path(image_path)
            if os.path.exists(confidence_path):
                confidence_img = Image.open(confidence_path).convert("L")
                confidence = self.transform(confidence_img)
            else:
                if cam_name not in self._missing_confidence_warned_cameras:
                    print(f"[Warning] Missing confidence map for {cam_name}, falling back to ones.")
                    self._missing_confidence_warned_cameras.add(cam_name)

        return rgb_tensor, alpha_mask, confidence, is_real_view

    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img, alpha_mask, confidence, is_real_view = self._load_image_alpha_confidence(self.image_paths[index])
        return img, self.image_poses[index], self.image_times[index], alpha_mask, confidence, is_real_view
    def load_pose(self,index):
        return self.image_poses[index]