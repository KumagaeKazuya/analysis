import json
import os
import argparse

# 保存するデータ (REFERENCE_MONITORS)
REFERENCE_MONITORS = [
    # 上段
    {"bbox": (269, 289, 151, 372), "name": "Monitor_0", "group": "top", "display_bbox": (261, 290, 365, 351)},
    {"bbox": (414, 288, 283, 372), "name": "Monitor_1", "group": "top", "display_bbox": (368, 289, 466, 347)},
    {"bbox": (569, 287, 417, 409), "name": "Monitor_2", "group": "top", "display_bbox": (474, 288, 585, 350)},
    {"bbox": (761, 282, 629, 372), "name": "Monitor_3", "group": "top", "display_bbox": (683, 287, 790, 350)},
    {"bbox": (891, 276, 788, 375), "name": "Monitor_4", "group": "top", "display_bbox": (800, 285, 902, 348)},
    {"bbox": (996, 274, 896, 368), "name": "Monitor_5", "group": "top", "display_bbox": (915, 283, 1017, 341)},
    {"bbox": (1136, 270, 1038, 367), "name": "Monitor_6", "group": "top", "display_bbox": (1037, 281, 1138, 345)},
    {"bbox": (1273, 276, 1177, 364), "name": "Monitor_7", "group": "top", "display_bbox": (1164, 282, 1256, 344)},
    {"bbox": (1397, 268, 1293, 360), "name": "Monitor_8", "group": "top", "display_bbox": (1282, 282, 1376, 344)},
    {"bbox": (1502, 269, 1636, 428), "name": "Monitor_9", "group": "top", "display_bbox": (1511, 275, 1596, 336)},
    {"bbox": (1621, 259, 1799, 352), "name": "Monitor_10", "group": "top", "display_bbox": (1620, 273, 1698, 334)},
    {"bbox": (1788, 255, 1912, 348), "name": "Monitor_11", "group": "top", "display_bbox": (1736, 274, 1802, 330)},
    
    # 中段
    {"bbox": (384, 372, 144, 614), "name": "Monitor_12", "group": "middle", "display_bbox": (285, 376, 429, 458)},
    {"bbox": (677, 369, 488, 553), "name": "Monitor_13", "group": "middle", "display_bbox": (564, 378, 704, 460)},
    {"bbox": (854, 370, 653, 559), "name": "Monitor_14", "group": "middle", "display_bbox": (713, 375, 854, 457)},
    {"bbox": (1026, 372, 845, 561), "name": "Monitor_15", "group": "middle", "display_bbox": (875, 375, 1019, 459)},
    {"bbox": (1046, 377, 1214, 565), "name": "Monitor_16", "group": "middle", "display_bbox": (1040, 372, 1179, 453)},
    {"bbox": (1213, 374, 1396, 558), "name": "Monitor_17", "group": "middle", "display_bbox": (1212, 376, 1342, 452)},
    {"bbox": (1380, 370, 1608, 548), "name": "Monitor_18", "group": "middle", "display_bbox": (1375, 367, 1511, 456)},
    {"bbox": (1677, 363, 1901, 613), "name": "Monitor_19", "group": "middle", "display_bbox": (1677, 361, 1779, 441)},
    
    # 下段
    {"bbox": (434, 605, 140, 1023), "name": "Monitor_20", "group": "bottom", "display_bbox": (319, 558, 534, 676)},
    {"bbox": (754, 601, 418, 1015), "name": "Monitor_21", "group": "bottom", "display_bbox": (531, 566, 758, 686)},
    {"bbox": (1005, 634, 676, 1035), "name": "Monitor_22", "group": "bottom", "display_bbox": (781, 570, 1003, 686)},
    {"bbox": (1246, 594, 1001, 1007), "name": "Monitor_23", "group": "bottom", "display_bbox": (1019, 569, 1224, 687)},
    {"bbox": (1265, 594, 1570, 1008), "name": "Monitor_24", "group": "bottom", "display_bbox": (1295, 567, 1474, 677)},
    {"bbox": (1531, 587, 1907, 1018), "name": "Monitor_25", "group": "bottom", "display_bbox": (1545, 563, 1700, 665)},
]

# ファイル名
file_name = "monitor_config.json"

def save_monitor_config_to_json(data, filename):
    """
    Pythonのリスト/辞書をJSONファイルとして保存する。
    ★ 正しい辞書形式で保存するように修正
    """
    try:
        # ★ モニターデータにIDを追加し、正しい形式で保存
        processed_monitors = []
        for i, monitor in enumerate(data):
            # タプルをリストに変換し、IDを追加
            processed_monitor = {
                "id": i,
                "name": monitor["name"],
                "group": monitor["group"],
                "bbox": list(monitor["bbox"]),  # タプル → リスト
                "display_bbox": list(monitor["display_bbox"]),  # タプル → リスト
                "reference_number": int(monitor["name"].split("_")[1])  # Monitor_0 → 0
            }
            processed_monitors.append(processed_monitor)
        
        # ★ 正しい辞書形式で保存
        config_data = {
            "monitors": processed_monitors,
            "total_monitors": len(processed_monitors),
            "config_used": {
                "method": "manual_reference",
                "source": "REFERENCE_MONITORS",
                "created_by": "12.2.step2.py"
            },
            "groups": {
                "top": [m for m in processed_monitors if m["group"] == "top"],
                "middle": [m for m in processed_monitors if m["group"] == "middle"],
                "bottom": [m for m in processed_monitors if m["group"] == "bottom"]
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 設定データはファイル '{filename}' に正常に保存されました。")
        print(f"📊 保存された情報:")
        print(f"   - 総モニター数: {len(processed_monitors)}台")
        print(f"   - 上段: {len(config_data['groups']['top'])}台")
        print(f"   - 中段: {len(config_data['groups']['middle'])}台")
        print(f"   - 下段: {len(config_data['groups']['bottom'])}台")
        print(f"   - 形式: 辞書形式 (monitors配列)")
        
    except IOError as e:
        print(f"❌ ファイルの書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")

# 関数を実行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モニター設定ファイル生成")
    parser.add_argument("--output-dir", type=str, required=True, help="プロジェクトフォルダ（kameoka1.pyで作成したもの）")
    args = parser.parse_args()

    json_dir = os.path.join(args.output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    file_name = os.path.join(json_dir, "monitor_config.json")

    print("=" * 60)
    print("モニター設定ファイル生成")
    print("=" * 60)
    save_monitor_config_to_json(REFERENCE_MONITORS, file_name)
    
    # 生成されたファイルを検証
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"\n✅ 検証完了: JSONファイルは有効です")
        print(f"   - キー: {list(test_data.keys())}")
        print(f"   - monitors配列: {len(test_data['monitors'])}件")
    except Exception as e:
        print(f"\n❌ 検証エラー: {e}")