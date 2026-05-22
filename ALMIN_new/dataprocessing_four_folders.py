import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import torch.utils.data as Data
import torch


def load_ir_image(path):
    img = imread(path).astype(np.float32) / 255.0  # 先读取并归一化
    if img.ndim == 3 and img.shape[2] == 3:
        # 彩色图像 → 强制转灰度
        img = rgb2gray(img)  # 输出形状: (H, W)
    elif img.ndim == 2:
        # 原本就是灰度图，不变
        pass
    else:
        raise ValueError(f"Unexpected IR image shape: {img.shape}")

    # 最后加一个通道维度，变成 (1, H, W)
    img = img[None, :, :]
    return img


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y


def Im2Patch(img, win, stride=1):
    endc, endw, endh = img.shape
    patch_w = (endw - win) // stride + 1
    patch_h = (endh - win) // stride + 1
    TotalPatNum = patch_w * patch_h
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    k = 0
    for i in range(0, endw - win + 1, stride):
        for j in range(0, endh - win + 1, stride):
            patch = img[:, i:i + win, j:j + win]  # 取单个patch，形状 (endc, win, win)
            Y[:, :, k] = patch.reshape(endc, -1)  # 拉平patch
            k += 1

    return Y.reshape([endc, win, win, TotalPatNum])


# 该函数用于判断一张图像是否为低对比度图像。
# 参数说明：
#   image: 输入的图像（可以是二维灰度图像或单通道图像数组）
#   fraction_threshold: 对比度阈值，默认0.1。若对比度低于该阈值，则认为是低对比度图像
#   lower_percentile: 计算像素强度下分位数的百分比，默认10（即10%分位点）
#   upper_percentile: 计算像素强度上分位数的百分比，默认90（即90%分位点）
# 实现思路：
#   1. 计算图像像素值的10%分位点和90%分位点，分别作为亮度的下限和上限（limits）。
#   2. 计算这两个分位点的差值与上限的比值（ratio），即对比度的一个度量。
#   3. 如果ratio小于fraction_threshold，则认为图像对比度过低，返回True，否则返回False。
def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """判断图像是否为低对比度图像。"""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


# 配置参数
data_name = "four_folders_data"
img_size = 128  # patch size
stride = 200  # patch stride

# 四个文件夹路径 - 请根据实际情况修改这些路径
IR_files = sorted(get_img_file(r"MSRS_train/ir"))                    # 红外图像文件夹
VIS_files = sorted(get_img_file(r"MSRS_train/vi"))                   # 可见光图像文件夹
ENHANCE_files = sorted(get_img_file(r"MSRS_train/illumination_features"))  # 增强图像文件夹
ILLUM_files = sorted(get_img_file(r"MSRS_train/illumination"))       # 光照图像文件夹

# 验证四个文件夹中的图片数量是否一致
assert len(IR_files) == len(VIS_files) == len(ENHANCE_files) == len(ILLUM_files), \
    f"图片数量不匹配: IR={len(IR_files)}, VIS={len(VIS_files)}, ENHANCE={len(ENHANCE_files)}, ILLUM={len(ILLUM_files)}"

print(f"找到图片数量: {len(IR_files)}")
print(f"IR文件夹: {len(IR_files)} 张图片")
print(f"VIS文件夹: {len(VIS_files)} 张图片") 
print(f"ENHANCE文件夹: {len(ENHANCE_files)} 张图片")
print(f"ILLUM文件夹: {len(ILLUM_files)} 张图片")

# 创建HDF5文件
h5f = h5py.File(os.path.join('./data',
                             data_name + '_imgsize_' + str(img_size) + "_stride_" + str(stride) + '.h5'),
                'w')

# 创建四个组来存储不同类型的patch
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
h5_enhance = h5f.create_group('enhance_patchs')
h5_illum = h5f.create_group('illum_patchs')

train_num = 0
total_patches = 0
filtered_patches = 0

for i in tqdm(range(len(IR_files))):
    # 读取四张图片
    I_VIS = imread(VIS_files[i]).astype(np.float32)[None, :, :] / 255.
    I_VIS = I_VIS[0].transpose(2, 0, 1)
    I_VIS = rgb2y(I_VIS)
    
    I_IR = load_ir_image(IR_files[i])
    I_ENHANCE = load_ir_image(ENHANCE_files[i])
    I_ILLUM = load_ir_image(ILLUM_files[i])

    # 将图片切分成patch
    I_IR_Patch_Group = Im2Patch(I_IR, img_size, stride)
    I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)
    I_ENHANCE_Patch_Group = Im2Patch(I_ENHANCE, img_size, stride)
    I_ILLUM_Patch_Group = Im2Patch(I_ILLUM, img_size, stride)

    # 处理每个patch
    for ii in range(I_IR_Patch_Group.shape[-1]):
        total_patches += 1
        
        # 检查四个patch的对比度
        bad_IR = is_low_contrast(I_IR_Patch_Group[0, :, :, ii])
        bad_VIS = is_low_contrast(I_VIS_Patch_Group[0, :, :, ii])
        bad_ENHANCE = is_low_contrast(I_ENHANCE_Patch_Group[0, :, :, ii])
        bad_ILLUM = is_low_contrast(I_ILLUM_Patch_Group[0, :, :, ii])

        # 只有当所有四个patch都不是低对比度时才保存
        if not (bad_IR or bad_VIS or bad_ENHANCE or bad_ILLUM):
            avl_IR = I_IR_Patch_Group[0, :, :, ii]
            avl_VIS = I_VIS_Patch_Group[0, :, :, ii]
            avl_ENHANCE = I_ENHANCE_Patch_Group[0, :, :, ii]
            avl_ILLUM = I_ILLUM_Patch_Group[0, :, :, ii]
            
            # 添加通道维度
            avl_IR = avl_IR[None, ...]
            avl_VIS = avl_VIS[None, ...]
            avl_ENHANCE = avl_ENHANCE[None, ...]
            avl_ILLUM = avl_ILLUM[None, ...]

            # 保存到HDF5文件
            h5_ir.create_dataset(str(train_num), data=avl_IR,
                                 dtype=avl_IR.dtype, shape=avl_IR.shape)
            h5_vis.create_dataset(str(train_num), data=avl_VIS,
                                  dtype=avl_VIS.dtype, shape=avl_VIS.shape)
            h5_enhance.create_dataset(str(train_num), data=avl_ENHANCE,
                                      dtype=avl_ENHANCE.dtype, shape=avl_ENHANCE.shape)
            h5_illum.create_dataset(str(train_num), data=avl_ILLUM,
                                    dtype=avl_ILLUM.dtype, shape=avl_ILLUM.shape)
            train_num += 1
        else:
            filtered_patches += 1

h5f.close()

# 打印统计信息
print(f"\n处理完成!")
print(f"总patch数: {total_patches}")
print(f"过滤掉的patch数: {filtered_patches}")
print(f"保留的patch数: {train_num}")
print(f"过滤率: {filtered_patches/total_patches*100:.2f}%")

# 验证生成的HDF5文件
with h5py.File(os.path.join('data',
                            data_name + '_imgsize_' + str(img_size) + "_stride_" + str(stride) + '.h5'), "r") as f:
    print(f"\nHDF5文件内容:")
    for key in f.keys():
        print(f"{f[key]}, {key}, {f[key].name}")


class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path

        h5f = h5py.File(h5file_path, 'r')

        # 检查必须存在的 key - 现在检查四个组
        required_keys = ['ir_patchs', 'vis_patchs', 'enhance_patchs', 'illum_patchs']
        for k in required_keys:
            if k not in h5f:
                raise KeyError(f"H5 文件中缺少必须的 key: '{k}'")

        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]

        # 加载四个模态的数据
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        ENHANCE = np.array(h5f['enhance_patchs'][key])
        ILLUM = np.array(h5f['illum_patchs'][key])

        h5f.close()
        
        # 返回四个张量：可见光、红外、增强、光照
        return torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(ENHANCE), torch.Tensor(ILLUM)


# 测试数据加载器
if __name__ == "__main__":
    # 测试数据加载
    h5_file_path = os.path.join('data', data_name + '_imgsize_' + str(img_size) + "_stride_" + str(stride) + '.h5')
    
    if os.path.exists(h5_file_path):
        print(f"\n测试数据加载器...")
        dataset = H5Dataset(h5_file_path)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试加载第一个样本
        if len(dataset) > 0:
            vis, ir, enhance, illum = dataset[0]
            print(f"VIS shape: {vis.shape}")
            print(f"IR shape: {ir.shape}")
            print(f"ENHANCE shape: {enhance.shape}")
            print(f"ILLUM shape: {illum.shape}")
            print("数据加载测试成功!")
        else:
            print("数据集为空!")
    else:
        print(f"H5文件不存在: {h5_file_path}")
