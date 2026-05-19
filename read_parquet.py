import pandas as pd
import sys

# Cài đặt pandas để hiển thị toàn bộ cột và hàng
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def read_parquet_head(file_path, n_rows=5):
    """
    Đọc file parquet và hiển thị n dòng đầu tiên
    
    Args:
        file_path (str): Đường dẫn đến file .parquet
        n_rows (int): Số dòng cần hiển thị (mặc định: 5)
    """
    try:
        # Đọc file parquet
        df = pd.read_parquet(file_path)
        
        print(f"File: {file_path}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\n" + "="*80 + "\n")
        
        # Hiển thị n dòng đầu tiên
        print(f"First {n_rows} rows:")
        print(df.head(n_rows))
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")

def get_unique_dates(file_path):
    """
    Đọc file parquet và thống kê tất cả các ngày khác nhau trong cột updated_date
    
    Args:
        file_path (str): Đường dẫn đến file .parquet
    """
    try:
        # Đọc file parquet
        df = pd.read_parquet(file_path)
        
        # Kiểm tra xem cột updated_date có tồn tại không
        if 'updated_date' not in df.columns:
            print("Error: Column 'updated_date' not found in the file")
            return
        
        # Chuyển đổi cột updated_date thành datetime
        df['updated_date'] = pd.to_datetime(df['updated_date'])
        
        # Trích xuất phần ngày (date only)
        df['date'] = df['updated_date'].dt.date
        
        # Lấy tất cả các ngày khác nhau và sắp xếp
        unique_dates = sorted(df['date'].unique())
        
        print(f"\n{'='*80}")
        print(f"THỐNG KÊ CÁC NGÀY KHÁC NHAU TRONG FILE")
        print(f"{'='*80}\n")
        
        print(f"Tổng số dòng dữ liệu: {len(df)}")
        print(f"Tổng số ngày khác nhau: {len(unique_dates)}\n")
        
        print("Danh sách các ngày:")
        print("-" * 40)
        for idx, date in enumerate(unique_dates, 1):
            count = (df['date'] == date).sum()
            print(f"{idx:3d}. {date} ({count:5d} records)")
        
        print("-" * 40)
        print(f"\nNgày sớm nhất: {unique_dates[0]}")
        print(f"Ngày muộn nhất: {unique_dates[-1]}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    
    file_path = r"C:\Users\Rmie\Desktop\Workspace\cs116\task1\data\transaction_full_2025.parquet"
    n_rows = 100
    
    # Hiển thị n dòng đầu tiên
    # read_parquet_head(file_path, n_rows)
    
    # Thống kê tất cả các ngày khác nhau
    # get_unique_dates(file_path)
    
    # Xuất dữ liệu ra file CSV
    try:
        df = pd.read_parquet(file_path)
        df_head = df.head(10)
        output_csv = r"C:\Users\Rmie\Desktop\Workspace\cs116\task1\data\transaction_2025.csv"
        df_head.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n{'='*80}")
        print(f"Xuất dữ liệu thành công!")
        print(f"File CSV: {output_csv}")
        print(f"Số dòng: {len(df_head)}, Số cột: {len(df_head.columns)}")
        print(f"{'='*80}")
    except Exception as e:
        print(f"Lỗi xuất file CSV: {e}")
