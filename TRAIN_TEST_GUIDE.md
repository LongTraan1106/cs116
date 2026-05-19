# Hướng Dẫn Train & Test Personalized Item Recommendation System

## 📋 Mục Lục
1. [Mô Tả Bài Toán](#mô-tả-bài-toán)
2. [Yêu Cầu Dữ Liệu](#yêu-cầu-dữ-liệu)
3. [Cài Đặt & Setup](#cài-đặt--setup)
4. [Quy Trình Training](#quy-trình-training)
5. [Quy Trình Testing](#quy-trình-testing)
6. [Hiểu Kết Quả](#hiểu-kết-quả)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Mô Tả Bài Toán

**Personalized Item Recommendation (PIR)**

### Dữ liệu
- **Training**: 1/1/2025 - 30/10/2025 (300 ngày)
- **Validation**: 31/10/2025 - 30/11/2025 (31 ngày)
- **Test (Private)**: 1/12/2025 - 31/12/2025 (31 ngày)

### Output
Dictionary mapping: `{customer_id: [item_1, item_2, ..., item_10]}`

### Metrics
- **Precision@10**: Tỷ lệ recommend đúng
- **Recall@10**: Coverage khách hàng
- **NDCG@10**: Ranking quality
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank (1/rank of first hit)
- **IoU**: Intersection over Union

### Cold Start Handling
- Khách hàng mới tài khoản trong 1/12-31/12 vẫn được tính
- Fallback: global trending items

---

## 📊 Yêu Cầu Dữ Liệu

### File Parquet (Bắt Buộc)
```
data/
├── items.parquet                    # Thông tin item
├── transaction_full_2025.parquet    # Lịch sử mua hàng
└── event_full_2025.parquet         # (Optional) View, ATC events
```

### Schema Yêu Cầu

#### `items.parquet`
```
Columns:
- item_id (String/Int)
- price (Float)
- category_l1, category_l2, category_l3, category (String)
- brand (String)
- manufacturer (String)
- description (String)
- sale_status, size (String)
```

#### `transaction_full_2025.parquet`
```
Columns:
- bill_id, customer_id, item_id (String/Int)
- price (Float)
- quantity (Int)
- event_type (String) = 'Purchase'
- updated_date (String, format: YYYY-MM-DD HH:MM:SS.mmm)
- location_name (String)
```

#### `event_full_2025.parquet` (Optional)
```
Columns:
- customer_id, item_id (String/Int)
- quantity (Int)
- event_type (String) = 'view_item', 'add-to-cart', 'Purchase'
- updated_date (String)
```

---

## 🔧 Cài Đặt & Setup

### 1. Cài Đặt Environment

#### Option A: Pip (Khuyến Nghị)
```bash
# Tạo virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# hoặc: source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

#### Option B: Conda
```bash
conda create -n recsys python=3.10
conda activate recsys
pip install -r requirements.txt
```

### 2. Kiểm Tra Dữ Liệu
```bash
# Đọc sample từ parquet
python read_parquet.py
```

Output mong đợi:
```
File: data/transaction_full_2025.parquet
Shape: XXXXX rows × X columns
First 5 rows:
...
```

### 3. Tạo Ground Truth (Test Set)
```bash
python create_groundtruth.py
```

Output:
```
>>> Creating Ground Truth for Test Set <<<
Loading transaction data from data/transaction_full_2025.parquet...
Available columns: [...]
✅ Created Ground Truth with XXX unique customers
   - Total unique items: XXXX
   - Average items per customer: X.XX
✅ Ground Truth saved successfully!
```

**Output file**: `data/groundtruth.pkl`

---

## 🚀 Quy Trình Training

### Pipeline Gồm 4 Giai Đoạn

```
┌──────────────────────────────────────────────┐
│  STAGE 1: CANDIDATE GENERATION               │
│  - CF models (Cosine, TFIDF)                │
│  - Co-occurrence matrices                    │
│  - Output: ~300-500 candidates per user      │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│  STAGE 2: FEATURE ENGINEERING & TRAINING     │
│  - Calculate 12 features per (user, item)   │
│  - LightGBM Ranker training                 │
│  - Hard negative sampling                    │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│  STAGE 3: INFERENCE & HYBRID RERANKING       │
│  - Generate predictions on candidates       │
│  - Apply business rules (price, brand, etc) │
│  - Top-10 per user                          │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│  STAGE 4: EXPORT & EVALUATION                │
│  - JSON export                              │
│  - Pickle export                            │
│  - Compute metrics (6 metrics)              │
└──────────────────────────────────────────────┘
```

### CF Method Hiện Tại

- CF dùng 2 mô hình item-based nearest neighbors: cosine và TF-IDF.
- Seed để sinh candidate không chỉ lấy purchase history mà còn lấy riêng `view_item` và `add_to_cart` theo từng `customer_id`.
- Thứ tự ưu tiên của seed signal:
   - purchase gần nhất: mạnh nhất
   - add_to_cart: mạnh thứ hai
   - view_item: nhẹ hơn
- Candidate score được cộng dồn theo:
   - số lần một item xuất hiện như neighbor từ nhiều seed khác nhau
   - mức ưu tiên của loại signal
   - độ gần của lịch sử gần nhất
- Item đã từng được user mua/xem/ATC vẫn bị loại khỏi candidate cuối cùng nếu đã nằm trong lịch sử train tương ứng.

### Chạy Training

```bash
python main.py
```

**Thời gian chạy**: ~30-60 phút (phụ thuộc dataset size)

**Logs quan trọng**:
```
>>> START PIPELINE >>>
>> Processing data...
-- TIME SPLIT DEBUG --
1. TEST RANGE:   (> 2025-12-01) --> (<= 2025-12-31)
2. VAL RANGE:    (> 2025-10-31) --> (<= 2025-11-30)
3. RECENT RANGE: (> 2025-10-01) --> (<= 2025-10-30)
4. HIST RANGE:   (> 2025-01-01) --> (<= 2025-10-30)

--- STAGE 1: CANDIDATE GENERATION ---
>> Generating new candidates...
>> Getting Trending Items...
>> Pre-computing Item Similarities...
>> Generating candidates for XXXXX customers...

--- STAGE 2: TRAINING MODEL ---
>> Positive Sampling...
>> Negative Sampling...
>> Starting Training...
>> Training Model: LightGBM (LightGBM Ranker)...

--- STAGE 3: INFERENCE & HYBRID RERANKING ---
>> Checking Inference files...
>> Sorting with Hybrid Logic...

--- STAGE 4: EXPORT & EVALUATION ---
>> Grouping items...
>> Streaming results to result.json...
✅ Exported JSON successfully.
>> Starting Evaluation...
```

### Kiểm Tra Tiến Độ

Các file tạm được lưu:
```
data/table/
├── temp_candidates/          # Stage 1 output
├── temp_features_train/      # Stage 2 features
└── temp_features_inference/  # Stage 3 features

candidates_stage1.parquet     # Cached candidates
lgbm_model.pkl               # Trained model
final_submission.parquet     # Final predictions
```

**Lưu ý**: Nếu chạy lại, xóa cache để chạy từ đầu:
```bash
# Windows
rmdir /s /q temp_* data\table\ candidates_stage1.parquet final_submission.parquet lgbm_model.pkl

# Linux/Mac
rm -rf temp_* data/table/ candidates_stage1.parquet final_submission.parquet lgbm_model.pkl
```

---

## 📈 Quy Trình Testing

Testing **tự động** diễn ra trong Stage 4 của main.py

### 1. Evaluation Metrics
```
─────────────────────────────────────────────────────
📊 FULL EVALUATION REPORT @ K=10
   Evaluated on XXX customers
─────────────────────────────────────────────────────
   Precision@10:    0.XXXX  (XX.XX%)
   Recall@10:       0.XXXX  (XX.XX%)
   NDCG@10:         0.XXXX  (XX.XX%)
   MAP (Mean Avg Prec):  0.XXXX  (XX.XX%)
   MRR (Recip Rank):     0.XXXX  (XX.XX%)
   IoU (Union/Intersection): 0.XXXX  (XX.XX%)
─────────────────────────────────────────────────────
```

### 2. Output Files

```
result.json                  # Dict: {customer_id: [items]}
result.pkl                   # Pickle backup
evaluation_report.json       # Metrics report
final_submission.parquet     # Raw predictions (debugging)
```

### 3. Kiểm Tra Kết Quả

#### Xem top recommendations cho 1 customer
```python
import json

with open("result.json", "r") as f:
    results = json.load(f)

customer_id = 123456
recommendations = results[str(customer_id)]
print(f"Recommendations for customer {customer_id}:")
print(recommendations)
```

#### Xem chi tiết metrics
```python
import json

with open("evaluation_report.json", "r") as f:
    metrics = json.load(f)

print(f"Precision@10: {metrics['precision@k']:.4f}")
print(f"Recall@10: {metrics['recall@k']:.4f}")
print(f"NDCG@10: {metrics['ndcg@k']:.4f}")
print(f"MAP: {metrics['map']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Evaluated on {metrics['n_users']} customers")
```

---

## 💡 Hiểu Kết Quả

### Metrics Giải Thích

| Metric | Công Thức | Ý Nghĩa | Target |
|--------|----------|---------|--------|
| **Precision@10** | hits / 10 | Tỷ lệ items recommend đúng | > 30% |
| **Recall@10** | hits / len(gt) | Khách hàng mua bao nhiêu % từ gợi ý | > 40% |
| **NDCG@10** | DCG / IDCG | Ranking quality (vị trí hit quan trọng) | > 50% |
| **MAP** | Σ(precision@k) / n_rel | Trung bình precision từ vị trí 1-10 | > 30% |
| **MRR** | 1 / rank(first_hit) | Độ sâu tìm hit đầu tiên | > 40% |
| **IoU** | \|A∩B\| / \|A∪B\| | Tương đồng recommend vs actual | > 20% |

### Đánh Giá Kết Quả

```
Excellent: Precision > 40%, NDCG > 60%, MAP > 40%
Good:      Precision > 30%, NDCG > 50%, MAP > 30%
Fair:      Precision > 20%, NDCG > 40%, MAP > 20%
Poor:      Precision < 20%
```

### Debug Tips

**Precision thấp (< 20%)**
- → Candidates quá nhiều không liên quan
- → Features không capture user preference
- → Model underfitting

**Recall thấp (< 30%)**
- → Top-10 không cover user purchase
- → Need more candidates, hoặc model predictions sai
- → Check SCORE_THRESHOLD (dòng 201 main.py)

**NDCG thấp (< 40%)**
- → Ranking order sai
- → Feature weights không optimal
- → Need tune LightGBM hyperparameters

---

## 🛠️ Troubleshooting

### Error 1: Data không tìm thấy
```
Error: data/items.parquet not found
```
**Fix**:
```bash
# Kiểm tra file tồn tại
ls -la data/

# Kiểm tra parquet format
python read_parquet.py
```

### Error 2: Date parsing error
```
Error: Cannot find date column. Available: [...]
```
**Fix**:
- Check `updated_date` format: `YYYY-MM-DD HH:MM:SS.mmm`
- Hoặc có cột `created_date` sẵn

### Error 3: Out of Memory
```
MemoryError: Unable to allocate ...
```
**Fix**:
```python
# Trong main.py, giảm batch_size
batch_size = 100000  # Thay vì 200000

# Hoặc giảm n_estimators
n_estimators=200  # Thay vì 500
```

### Error 4: Ground truth không khớp
```
⚠️ Không có user nào trùng khớp!
```
**Fix**:
- GT phải từ test period (1/12-31/12)
- Transaction phải có `event_type == 'Purchase'`
- Check `create_groundtruth.py` logic

### Error 5: LightGBM warning
```
UserWarning: Found only ... unique labels
```
**Fix**: Chuẩn bị dữ liệu train:
```python
# Đảm bảo đủ positives & negatives
pos_count = len(df_train[df_train['target'] == 1])
neg_count = len(df_train[df_train['target'] == 0])
print(f"Positives: {pos_count}, Negatives: {neg_count}")
# Target: pos/neg ratio ~ 1:10
```

---

## 📱 Bonus: Streamlit Dashboard

Xem recommendations interactively:
```bash
streamlit run app.py
```

**Features**:
- 🔍 Tìm kiếm customer
- 👤 Xem lịch sử mua hàng
- 🎯 Top 10 recommendations
- 📊 Feature weights
- ✅ Hit analysis

---

## 📋 Checklist Hoàn Tất

- [ ] Cài đặt dependencies: `pip install -r requirements.txt`
- [ ] Kiểm tra dữ liệu: `python read_parquet.py`
- [ ] Tạo GT: `python create_groundtruth.py`
- [ ] Chạy training: `python main.py`
- [ ] Kiểm tra result.json
- [ ] Kiểm tra evaluation_report.json
- [ ] (Optional) Chạy dashboard: `streamlit run app.py`

---

## 📞 Support

Nếu có lỗi:
1. Kiểm tra logs trong console
2. Check Troubleshooting section
3. Verify data format: `read_parquet.py`
4. Xem file cache: `temp_*.parquet`

---

**Last Updated**: 2026-05-12  
**Model**: LightGBM Ranker (Hybrid CF + Supervised Learning)  
**Data Period**: 2025 (Train+Val) vs Test (Private)
