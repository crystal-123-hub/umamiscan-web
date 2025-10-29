# predict.py
import torch
import numpy as np
from Bio import SeqIO
from io import StringIO
import pandas as pd
from typing import List, Tuple
import os


from model import ToxiPep_Model
from dataset import convert_to_graph_channel


from transformers import AutoTokenizer, EsmModel

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


d_model = 1280
d_ff = 640
n_layers = 4
n_heads = 4
max_len = 20
structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 20,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

# 预训练 ESM 模型名称
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# 缓存 tokenizer 和 model
_esm_tokenizer = None
_esm_model = None

def get_esm_model():
    global _esm_tokenizer, _esm_model
    if _esm_tokenizer is None:
        print("Loading ESM-2 tokenizer and model...")
        cache_dir = "/var/models/esm_cache"
        os.makedirs(cache_dir, exist_ok=True)

        _esm_tokenizer = AutoTokenizer.from_pretrained(
            ESM_MODEL_NAME,
            cache_dir=cache_dir
        )
        _esm_model = EsmModel.from_pretrained(
            ESM_MODEL_NAME,
            cache_dir=cache_dir
        ).to(device).eval()
    return _esm_tokenizer, _esm_model

# =============================
# 数据集类
# =============================
class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, esm_embeddings, graph_features):
        self.esm_embeddings = esm_embeddings
        self.graph_features = graph_features

    def __len__(self):
        return len(self.esm_embeddings)

    def __getitem__(self, idx):
        return (
            self.esm_embeddings[idx].to(torch.float32),
            self.graph_features[idx].to(torch.float32)
        )

# =============================
# 预测函数
# =============================
@torch.no_grad()
def predict(model, dataloader, device) -> Tuple[List[int], List[float]]:
    model.eval()
    predictions = []
    probabilities = []

    for batch_esm, batch_graph in dataloader:
        batch_esm = batch_esm.to(device)
        batch_graph = batch_graph.to(device)

        outputs = model(batch_esm, batch_graph, device)
        probs = torch.sigmoid(outputs).squeeze(-1)
        preds = (probs > 0.5).long()

        predictions.extend(preds.cpu().numpy().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())

    return predictions, probabilities


def run_prediction(fasta_text: str, output_file: str = "results/umamiscan_results.csv") -> Tuple[str, str]:
    """
    输入 FASTA 字符串，返回可展示的结果文本和保存的文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Parsing FASTA input...")
    fasta_io = StringIO(fasta_text)
    seq_ids = []
    sequences = []

    for r in SeqIO.parse(fasta_io, "fasta"):
        seq_id = r.id if r.id else f"seq_{len(seq_ids)}"
        seq_str = str(r.seq).upper()
        seq_str = seq_str.replace('U', 'C').replace('J', 'L').replace('O', 'K')
        seq_str = ''.join([aa if aa in 'ACDEFGHIKLMNPQRSTVWY' else 'X' for aa in seq_str])
        seq_ids.append(seq_id)
        sequences.append(seq_str)

    if len(sequences) == 0:
        raise ValueError("No valid sequences found in input.")

    print(f"Loaded {len(sequences)} sequences.")

    # Step 1: Generate ESM embeddings using ESM-2
    print("Generating ESM-2 embeddings...")
    tokenizer, esm_model = get_esm_model()
    esm_embs = []

    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=100).to(device)
            outputs = esm_model(**inputs)
            mean_emb = outputs.last_hidden_state.mean(1).cpu()  # [1, d_model]
            mean_emb = torch.nn.functional.pad(mean_emb, (0, 0, 0, max_len - 1))[:max_len]  # Pad to max_len
            esm_embs.append(mean_emb.squeeze(0))

    # Step 2: Generate graph features
    print("Generating graph structure features...")
    graph_features = [torch.tensor(convert_to_graph_channel(seq), dtype=torch.float32) for seq in sequences]

    # Step 3: DataLoader
    dataset = PeptideDataset(esm_embs, graph_features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Step 4: Load trained model
    print("Loading ToxiPep_Model...")
    model = ToxiPep_Model(
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        structural_config=structural_config
    ).to(device)

    best_model_path = "../ToxiPep-main/Code/best_model_mcc.pth"
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model weights not found: {best_model_path}")

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # Step 5: Predict
    print("Running inference...")
    preds, probs = predict(model, dataloader, device)

    # Step 6: Save results
    print(f"Saving to {output_file}...")
    results = []
    for i, (seq_id, seq, pred, prob) in enumerate(zip(seq_ids, sequences, preds, probs)):
        label = "Umami" if pred == 1 else "Non-umami"
        results.append({
            "ID": seq_id,
            "Sequence": seq,
            "Prediction": label,
            "Probability": round(prob, 6)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    df.to_csv(output_file.replace(".csv", ".txt"), sep="\t", index=False)

    # 返回表格文本用于前端显示
    result_text = df.to_string(index=False)

    return result_text, output_file

# =============================
# 命令行接口（兼容原逻辑）
# =============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ToxiPep Prediction with Precomputed ESM")
    parser.add_argument("-i", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--esm", type=str, required=True, help="Path to precomputed ESM embeddings (.pt)")
    parser.add_argument("-o", type=str, required=True, help="Output result file (txt/csv)")
    args = parser.parse_args()

    # 这里仍走原始逻辑（需要 .pt 文件）
    # （你可以选择是否保留这部分，或统一走上面的 run_prediction）
    raise NotImplementedError("Command-line mode requires precomputed ESM embeddings. Use web version for auto-embedding.")