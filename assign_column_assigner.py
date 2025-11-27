import pandas as pd
import os
import re

def main():
    csv_path = input("対象CSVファイル名を入力してください: ").strip()
    df = pd.read_csv(csv_path)
    df['column_position'] = None

    # フレーム番号抽出（frame列が '...frameXX...' 形式の場合）
    if 'frame' not in df.columns:
        print("frame列が見つかりません")
        return

    # frame番号を抽出して新しい列に
    df['frame_num'] = df['frame'].apply(
        lambda x: int(re.search(r'frame(\d+)', str(x)).group(1)) if re.search(r'frame(\d+)', str(x)) else -1
    )
    # 番号で昇順ソート
    df = df.sort_values('frame_num')

    frames = df['frame'].unique()
    print(f"全{len(frames)}フレームを番号順で処理します")

    # frame_numで昇順ループ
    for frame_num in sorted(df['frame_num'].unique()):
        frame_df = df[df['frame_num'] == frame_num]
        ids = frame_df['person_id'].unique()
        print(f"\nframe{frame_num}のID一覧: {list(ids)}")
        print("このフレームのIDごとに列番号を指定してください")
        print("例: --col1 a b --col2 c d --col3 e f")
        col_input = input("列指定: ").strip()

        # 入力例: --col1 a b --col2 c d --col3 e f
        col_map = {}
        for part in col_input.split('--'):
            if not part.strip():
                continue
            col_name, *members = part.strip().split()
            col_num = int(col_name.replace('col', ''))
            for member in members:
                col_map[member] = col_num

        # person_idごとにcolumn_positionを割り当て
        for idx in frame_df.index:
            pid = str(df.at[idx, 'person_id'])
            if pid in col_map:
                df.at[idx, 'column_position'] = col_map[pid]
            else:
                print(f"ID {pid} は指定されていません（frame{frame_num}）")

    # 保存先は元CSVと同じフォルダ
    out_path = os.path.join(
        os.path.dirname(csv_path),
        os.path.basename(csv_path).replace('.csv', '_with_column.csv')
    )
    # frame_num列は不要なので削除
    df.drop(columns=['frame_num'], inplace=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n保存完了: {out_path}")

if __name__ == "__main__":
    main()