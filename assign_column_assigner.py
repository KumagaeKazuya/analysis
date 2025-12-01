import pandas as pd
import os

def main():
    csv_path = input("対象CSVファイル名を入力してください: ").strip()
    df = pd.read_csv(csv_path)
    df['column_position'] = None

    # 列数の入力
    num_columns = int(input("何列まで割り当てますか？（例: 3）: ").strip())
    y_ranges = []
    for i in range(num_columns):
        y_min = float(input(f"{i+1}列目のy_min（下端最小値）を入力: ").strip())
        y_max = float(input(f"{i+1}列目のy_max（下端最大値）を入力: ").strip())
        y_ranges.append((y_min, y_max))

    # 各行のバウンディングボックス下端y2で列番号を自動割り当て
    for idx, row in df.iterrows():
        y2 = row['y2']
        assigned = False
        for col_num, (y_min, y_max) in enumerate(y_ranges, start=1):
            if y_min <= y2 < y_max:
                df.at[idx, 'column_position'] = col_num
                assigned = True
                break
        # 属さない場合はNone（何もしない）

    out_path = os.path.join(
        os.path.dirname(csv_path),
        os.path.basename(csv_path).replace('.csv', '_with_column.csv')
    )
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n保存完了: {out_path}")

if __name__ == "__main__":
    main()