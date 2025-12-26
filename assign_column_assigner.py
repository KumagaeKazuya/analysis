import pandas as pd
import os
def parse_id_list(input_str):
    """
    カンマ区切りのIDリストをintのリストに変換
    例: "0,2,5" → [0,2,5]
    """
    return [int(i.strip()) for i in input_str.split(",") if i.strip()]

def main():
    csv_path = input("対象CSVファイル名を入力してください: ").strip()
    df = pd.read_csv(csv_path)
    df['column_position'] = None

    column_ids = {}
    for col_num in range(1, 4):
        ids_str = input(f"カメラから{col_num}列目のperson_idをカンマ区切りで入力してください: ").strip()
        column_ids[col_num] = parse_id_list(ids_str)

    for idx, row in df.iterrows():
        id_val = row['person_id']
        for col_num, id_list in column_ids.items():
            if id_val in id_list:
                df.at[idx, 'column_position'] = col_num
                break

    out_path = os.path.join(
        os.path.dirname(csv_path),
        os.path.basename(csv_path).replace('.csv', '_with_column.csv')
    )
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n保存完了: {out_path}")

if __name__ == "__main__":
    main()