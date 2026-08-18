import os
import gc
import hashlib
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
import traceback

# 🌟 开启 TF32 矩阵乘法加速 (RTX 30/40 系显卡必备)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================================================
# 0. 核心模块：纯 PyTorch 几何特征离线提取器 (兼容版)
# =========================================================
class PeptideGeometryExtractor:
    def __init__(self):
        self.C_C_BOND_LENGTH = 1.54
        
    def extract_features(self, positions):
        # 1. 提取主链和真实侧链原子
        pos_N = positions[:, 0, :]   
        pos_CA = positions[:, 1, :]  
        pos_C = positions[:, 2, :]   
        
        # 2. 计算真实或虚拟的 C_beta 坐标
        pos_CB = self._compute_virtual_cb(pos_N, pos_CA, pos_C, positions)
        
        # 3. 计算侧链方向向量 (C_beta - C_alpha)
        vec_sidechain = pos_CB - pos_CA  
        
        # 4. 计算主链二面角 (Phi, Psi) 并进行正余弦编码
        dihedrals = self._compute_dihedrals(pos_N, pos_CA, pos_C) 
        dihedral_embeddings = torch.cat([
            torch.sin(dihedrals), 
            torch.cos(dihedrals)
        ], dim=-1) 
        
        return {
            "pos_CA": pos_CA,                      
            "vec_sidechain": vec_sidechain,        
            "dihedrals": dihedral_embeddings       
        }

    def _compute_virtual_cb(self, N, CA, C, full_positions):
        v_N = N - CA
        v_C = C - CA
        normal_vec = torch.cross(v_N, v_C, dim=-1)
        normal_vec = F.normalize(normal_vec, p=2, dim=-1, eps=1e-8)
        virtual_CB = CA + normal_vec * self.C_C_BOND_LENGTH
        
        real_CB = full_positions[:, 4, :]
        mask_has_cb = (torch.norm(real_CB, dim=-1) > 1e-3).unsqueeze(-1)
        final_CB = torch.where(mask_has_cb, real_CB, virtual_CB)
        return final_CB

    def _compute_dihedrals(self, N, CA, C):
        L = N.shape[0]
        # 显式继承输入张量的设备，由于后续已转 CPU，这里也会在 CPU 初始化
        angles = torch.zeros((L, 2), dtype=torch.float32, device=N.device)
        if L < 3: return angles
            
        def calculate_torsion(p0, p1, p2, p3):
            b0 = -1.0 * (p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            b1_normalized = F.normalize(b1, p=2, dim=-1, eps=1e-8)
            v = b0 - torch.sum(b0 * b1_normalized, dim=-1, keepdim=True) * b1_normalized
            w = torch.cross(b1_normalized, b2, dim=-1)
            x = torch.sum(v * b2, dim=-1)
            y = torch.sum(torch.cross(b1_normalized, v, dim=-1) * b2, dim=-1)
            return torch.atan2(y, x)

        phi = calculate_torsion(C[:-1], N[1:], CA[1:], C[1:])
        angles[1:, 0] = phi
        
        psi = calculate_torsion(N[:-1], CA[:-1], C[:-1], N[1:])
        angles[:-1, 1] = psi
        return angles

# =========================================================
# 1. 路径与离线环境 (已修改为相对路径)
# =========================================================
# 设定当前脚本所在目录即为 data 目录的基础路径
BASE_DIR = "."  
MODEL_DIR = os.path.join(BASE_DIR, "esmfold_v1")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

COORDS_DIR = os.path.join(BASE_DIR, "tensors") 
CSV_MAP_PATH = os.path.join(BASE_DIR, "sequence_to_tensor_mapping.csv")

os.makedirs(COORDS_DIR, exist_ok=True)
print("✅ 离线模式已启用，模型路径:", MODEL_DIR)

# =========================================================
# 2. 数据准备与长度过滤
# =========================================================
# 使用相对路径读取 CSV 文件
raw_df = pd.read_csv(os.path.join(BASE_DIR, 'peptides-complete-new_processed.csv'))
raw_df = raw_df[raw_df['TARGET ACTIVITY - TARGET SPECIES'] == "escherichia coli"]

# 使用 C 语言级别的过滤速度
valid_rows = raw_df['SEQUENCE'].dropna().str.fullmatch(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
raw_df = raw_df[valid_rows]
raw_df = raw_df[(raw_df['SEQUENCE'].str.len() >= 5) & (raw_df['SEQUENCE'].str.len() <= 400)]

unique_sequences = raw_df['SEQUENCE'].astype(str).unique().tolist()
unique_sequences.sort(key=len)
print(f"✅ 过滤异常长度后，大肠杆菌总计需处理序列数: {len(unique_sequences)}")

# =========================================================
# 3. 历史记录与断点续传 
# =========================================================
processed_hashes = set()
if os.path.exists(CSV_MAP_PATH):
    try:
        existing_df = pd.read_csv(CSV_MAP_PATH)
        processed_hashes = set(existing_df['hash_filename'].str.replace('.pt', '', regex=False))
        print(f"🔄 检测到历史记录！已恢复 {len(processed_hashes)} 条已完成的映射数据。")
    except Exception as e:
        print(f"⚠️ 读取历史映射表失败，报错: {e}")
else:
    with open(CSV_MAP_PATH, "w", encoding="utf-8") as f:
        f.write("sequence,hash_filename\n")

# =========================================================
# 4. 模型与提取器加载
# =========================================================
print("加载 ESMFold 与几何提取器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = EsmForProteinFolding.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, local_files_only=True).to("cuda")
model.eval()

# 实例化我们刚才写的几何特征提取器
geo_extractor = PeptideGeometryExtractor()

# =========================================================
# 5. 逐条处理与存盘 (显存管理与精度修复版)
# =========================================================
print("开始生成轻量化 3D 几何特征张量...")

with open(CSV_MAP_PATH, "a", encoding="utf-8") as csv_file:
    with torch.no_grad(): # 全局禁用梯度
        for i, seq in enumerate(tqdm(unique_sequences, desc="提取与精简")):
            seq_hash = hashlib.md5(seq.encode('utf-8')).hexdigest()
            save_path = os.path.join(COORDS_DIR, f"{seq_hash}.pt")

            if seq_hash in processed_hashes and os.path.exists(save_path):
                continue

            try:
                inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False).to("cuda")
                outputs = model(**inputs)
                
                # 从 GPU 转移到 CPU，释放显存准备做高精度几何计算
                positions = outputs.positions[0].cpu()
                plddt = outputs.plddt[0].cpu()

                if torch.isnan(positions).any() or torch.isinf(positions).any():
                    continue

                # 核心：在 CPU 内存中直接完成几何特征运算，防止 GPU 的 TF32 带来运算误差
                geo_features = geo_extractor.extract_features(positions)
                
                # 由于此时张量已经在 CPU 上，仅调用 .clone() 防止底层的巨大显存张量引用泄露
                final_data = {
                    "sequence": seq,
                    "plddt_CA": plddt[:, 1].clone(),
                    "pos_CA": geo_features["pos_CA"].clone(),
                    "vec_sidechain": geo_features["vec_sidechain"].clone(),
                    "dihedrals": geo_features["dihedrals"].clone()
                }
                
                torch.save(final_data, save_path)

                csv_file.write(f"{seq},{seq_hash}.pt\n")
                
                # 降低 flush 频率，减少硬盘 I/O 阻塞
                if i % 20 == 0:
                    csv_file.flush() 
                processed_hashes.add(seq_hash)

                # 利用 Python 引用计数机制，让 PyTorch 的 Allocator 自动复用显存
                del inputs, outputs, positions, plddt, geo_features, final_data

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️ 显存不足，跳过序列 {seq[:15]}...")
                    # 仅在真正 OOM 时清空缓存
                    torch.cuda.empty_cache() 
                    continue
                else:
                    print(f"\n❌ 序列 {seq[:15]}... 发生未知运行时错误: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"\n❌ 序列 {seq[:15]}... 出错: {e}")
                traceback.print_exc()

print(f"\n✅ 几何特征张量提取完毕！保存在: {COORDS_DIR}")