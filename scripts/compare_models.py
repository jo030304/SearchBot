"""
학습된 모델들의 결과 비교
사용법: python scripts/compare_models.py
"""

import sys
import json
from pathlib import Path
import argparse

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# matplotlib 선택적 import
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 사용
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib이 없습니다. 그래프는 생성되지 않습니다.")
    print("   설치: pip install matplotlib")

# pandas 선택적 import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas가 없습니다. 표는 간단한 형식으로 출력됩니다.")
    print("   설치: pip install pandas")

from src.models.backbones import MODEL_INFO


def load_histories(checkpoint_dir='checkpoints'):
    """모든 모델의 학습 히스토리 로드"""
    
    checkpoint_dir = Path(checkpoint_dir)
    histories = {}
    
    print("\n📂 학습 히스토리 로드 중...")
    print("-"*80)
    
    for model_name in MODEL_INFO.keys():
        history_path = checkpoint_dir / f'{model_name}_history.json'
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                histories[model_name] = json.load(f)
            print(f"  ✅ {model_name}")
        else:
            print(f"  ⏭️  {model_name} (미학습)")
    
    return histories


def compare_results(checkpoint_dir='checkpoints', output_dir='results'):
    """모델 결과 비교 및 저장"""
    
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 모델 성능 비교")
    print("="*80)
    
    # 히스토리 로드
    histories = load_histories(checkpoint_dir)
    
    if not histories:
        print("\n⚠️  학습 결과가 없습니다.")
        print("   먼저 다음 명령어를 실행하세요:")
        print("   python scripts/train.py --model resnet18 --epochs 30")
        return
    
    # 결과 수집
    results = []
    
    print("\n📈 모델별 성능:")
    print("-"*80)
    
    for model_name, history in histories.items():
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        results.append({
            'Model': MODEL_INFO[model_name]['name'],
            'Params': MODEL_INFO[model_name]['params'],
            'Best Val Acc': f"{best_val_acc:.2f}%",
            'Best Epoch': best_epoch,
            'Final Train Acc': f"{final_train_acc:.2f}%",
            'Final Val Acc': f"{final_val_acc:.2f}%",
            'Final Train Loss': f"{final_train_loss:.4f}",
            'Final Val Loss': f"{final_val_loss:.4f}",
            'Description': MODEL_INFO[model_name]['description']
        })
        
        print(f"  {MODEL_INFO[model_name]['name']:20s}: Best Val Acc = {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # 테이블 출력
    print("\n" + "="*80)
    print("📋 상세 비교표")
    print("="*80)
    
    if PANDAS_AVAILABLE:
        # pandas로 깔끔한 표 출력
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        
        # CSV 저장
        csv_path = output_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 CSV 저장: {csv_path}")
    else:
        # pandas 없이 수동 출력
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['Model']}:")
            for key, value in result.items():
                if key != 'Model':
                    print(f"     {key:20s}: {value}")
        
        # 수동 CSV 저장
        csv_path = output_dir / 'model_comparison.csv'
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            # 헤더
            headers = list(results[0].keys())
            f.write(','.join(headers) + '\n')
            # 데이터
            for result in results:
                row = [str(result[h]) for h in headers]
                f.write(','.join(row) + '\n')
        print(f"\n💾 CSV 저장: {csv_path}")
    
    # 그래프 생성
    if MATPLOTLIB_AVAILABLE:
        plot_comparison(histories, output_dir)
    else:
        print("\n⚠️  matplotlib이 없어 그래프를 생성할 수 없습니다.")
        print("   설치하려면: pip install matplotlib")
    
    print("\n" + "="*80)


def plot_comparison(histories, output_dir):
    """학습 곡선 비교 그래프 생성"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Loss 비교
    for idx, (model_name, history) in enumerate(histories.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors[idx % len(colors)]
        
        axes[0].plot(epochs, history['train_loss'], 
                    label=f"{MODEL_INFO[model_name]['name']} (Train)", 
                    linestyle='-', linewidth=2, alpha=0.8, color=color)
        axes[0].plot(epochs, history['val_loss'], 
                    label=f"{MODEL_INFO[model_name]['name']} (Val)", 
                    linestyle='--', linewidth=2, alpha=0.8, color=color)
    
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Accuracy 비교
    for idx, (model_name, history) in enumerate(histories.items()):
        epochs = range(1, len(history['train_acc']) + 1)
        color = colors[idx % len(colors)]
        
        axes[1].plot(epochs, history['train_acc'], 
                    label=f"{MODEL_INFO[model_name]['name']} (Train)", 
                    linestyle='-', linewidth=2, alpha=0.8, color=color)
        axes[1].plot(epochs, history['val_acc'], 
                    label=f"{MODEL_INFO[model_name]['name']} (Val)", 
                    linestyle='--', linewidth=2, alpha=0.8, color=color)
    
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 저장
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 그래프 저장: {plot_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='모델 결과 비교')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='체크포인트 디렉토리 (기본: checkpoints)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='결과 저장 디렉토리 (기본: results)')
    
    args = parser.parse_args()
    
    try:
        # 비교 실행
        compare_results(args.checkpoint_dir, args.output_dir)
        
        # 결과 확인 안내
        print("\n💡 생성된 파일:")
        print(f"  📄 CSV: {args.output_dir}/model_comparison.csv")
        if MATPLOTLIB_AVAILABLE:
            print(f"  📊 그래프: {args.output_dir}/training_curves.png")
        
        print("\n💡 파일 열기:")
        print(f"  open {args.output_dir}/model_comparison.csv")
        if MATPLOTLIB_AVAILABLE:
            print(f"  open {args.output_dir}/training_curves.png")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 해결 방법:")
        print("  1. 최소 1개 모델 학습 완료 확인")
        print("  2. checkpoints/ 폴더에 *_history.json 파일 확인")
        print("  3. 프로젝트 루트에서 실행")


if __name__ == "__main__":
    main()