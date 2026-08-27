import os
import random
import shutil
from collections import defaultdict


def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将指定格式的数据集划分为训练集、验证集和测试集。

    Args:
        dataset_dir (str): 数据集所在的根目录。
        train_ratio (float): 训练集所占比例。
        val_ratio (float): 验证集所占比例。
        test_ratio (float): 测试集所占比例。
    """
    # 确保比例总和为1
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("训练、验证和测试集的比例总和必须为1")

    # 1. 获取所有唯一的样本ID
    all_files = os.listdir(dataset_dir)
    # 通过查找 .pkl 文件来获取所有唯一的样本ID
    sample_ids = sorted([f.split('.')[0] for f in all_files if f.endswith('.pkl')])

    # 打印找到的样本总数
    print(f"总共找到 {len(sample_ids)} 个样本。")

    # 2. 随机打乱样本ID
    random.seed(42)  # 设置随机种子以保证结果可复现
    random.shuffle(sample_ids)

    # 3. 按比例计算各个集合的大小
    total_samples = len(sample_ids)
    train_split_index = int(total_samples * train_ratio)
    val_split_index = int(total_samples * (train_ratio + val_ratio))

    # 4. 切分ID列表
    train_ids = sample_ids[:train_split_index]
    val_ids = sample_ids[train_split_index:val_split_index]
    test_ids = sample_ids[val_split_index:]

    print(f"划分结果：")
    print(f" - 训练集样本数: {len(train_ids)}")
    print(f" - 验证集样本数: {len(val_ids)}")
    print(f" - 测试集样本数: {len(test_ids)}")

    # 5. 根据ID生成完整的文件列表
    file_lists = {
        'train': [],
        'val': [],
        'test': []
    }

    sets = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    # 定义每个样本ID对应的文件后缀
    file_suffixes = ['.pkl', '_origin_left.jpg', '_origin_right.jpg', '_seg.jpg']

    for set_name, ids in sets.items():
        for sample_id in ids:
            for suffix in file_suffixes:
                filename = f"{sample_id}{suffix}"
                if filename in all_files:
                    file_lists[set_name].append(filename)

    # 打印每个集合中的一些文件名作为示例
    print("\n--- 文件列表示例 ---")
    print("训练集文件 (前5个):", file_lists['train'][:5])
    print("验证集文件 (前5个):", file_lists['val'][:5])
    print("测试集文件 (前5个):", file_lists['test'][:5])

    # 6. (可选) 将文件移动到新的目录结构中
    # 如果你想将文件物理移动到 train/val/test 文件夹，可以取消下面的注释

    print("\n开始移动文件...")
    for set_name, files in file_lists.items():
        output_dir = os.path.join(dataset_dir, set_name)
        os.makedirs(output_dir, exist_ok=True)
        for filename in files:
            source_path = os.path.join(dataset_dir, filename)
            destination_path = os.path.join(output_dir, filename)
            shutil.move(source_path, destination_path)
    print("文件移动完成！")

    return file_lists


# --- 使用方法 ---
if __name__ == "__main__":
    # *****************************************************
    # ***** 请将此路径修改为你的数据集所在的文件夹路径 *****
    # *****************************************************
    dataset_directory = './save_path_new'

    # 假设你的数据集文件夹不存在，我们创建一个虚拟的
    if not os.path.exists(dataset_directory):
        print(f"警告：目录 '{dataset_directory}' 不存在。将创建一个虚拟数据集用于演示。")
        os.makedirs(dataset_directory)
        # 创建一些示例文件
        for i in range(100):
            base_name = f"{i:06d}"
            open(os.path.join(dataset_directory, f"{base_name}.pkl"), 'a').close()
            open(os.path.join(dataset_directory, f"{base_name}_origin_left.jpg"), 'a').close()
            open(os.path.join(dataset_directory, f"{base_name}_origin_right.jpg"), 'a').close()
            open(os.path.join(dataset_directory, f"{base_name}_seg.jpg"), 'a').close()

    # 执行划分
    divided_file_lists = split_dataset(dataset_directory)

    # 你现在可以根据 divided_file_lists['train'], divided_file_lists['val'],
    # divided_file_lists['test'] 来加载你的数据了。