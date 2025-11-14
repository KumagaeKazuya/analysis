#!/bin/bash
echo "🔍 実行監視開始: $(date)"
while true; do
    # Pythonプロセス確認
    python_pids=$(pgrep -f "improved_main.py")
    
    if [ -n "$python_pids" ]; then
        echo "⚡ $(date '+%H:%M:%S') - improved_main.py 実行中"
        
        for pid in $python_pids; do
            # メモリ使用量
            if [ -f "/proc/$pid/status" ]; then
                memory_kb=$(grep VmRSS /proc/$pid/status | awk '{print $2}')
                memory_mb=$((memory_kb / 1024))
                echo "  📊 PID:$pid Memory:${memory_mb}MB"
            fi
            
            # 開いているファイル（macOS/Linux対応）
            if command -v lsof >/dev/null; then
                pt_files=$(lsof -p $pid 2>/dev/null | grep "\.pt$" | awk '{print $9}')
                if [ -n "$pt_files" ]; then
                    echo "  🎯 使用モデル:"
                    echo "$pt_files" | while read file; do
                        if [ -f "$file" ]; then
                            size_mb=$(python3 -c "import os; print(f'{os.path.getsize('$file')/1024/1024:.1f}')" 2>/dev/null)
                            echo "    $file (${size_mb}MB)"
                        fi
                    done
                fi
            fi
        done
    else
        echo "⏳ $(date '+%H:%M:%S') - improved_main.py 待機中"
    fi
    
    sleep 5
done
