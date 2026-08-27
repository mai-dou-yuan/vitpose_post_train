import os
import glob
import pickle
from tqdm import tqdm
from collections import Counter

def check_user_distribution(save_path):
    print(f"正在分析生成的每一帧数据来源: {save_path}")
    pkl_files = glob.glob(os.path.join(save_path, "*.pkl"))
    
    user_counter = Counter()
    
    for pkl_path in tqdm(pkl_files, desc="读取PKL"):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                user_name = data.get('user', 'Unknown')
                user_counter[user_name] += 1
        except Exception as e:
            print(f"读取错误 {pkl_path}: {e}")

    print("\n=== 数据分布统计 ===")
    if not user_counter:
        print("没有找到任何有效数据。")
        total_frames = 0
    else:
        total_frames = sum(user_counter.values())
        # 按生成的数量降序打印
        for user, count in user_counter.most_common():
            print(f"用户: {user:<15} -> 贡献帧数: {count}")
    
    print(f"\n📊 数据集总帧数: {total_frames}")

if __name__ == "__main__":
    save_path = 'save_path_11'
    check_user_distribution(save_path)