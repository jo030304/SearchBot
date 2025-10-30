"""
데이터셋 분석 스크립트
사용법: python scripts/analyze_dataset.py
"""

import sys
import cv2
import json
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_dataset(root_dir='datasets'):
    """데이터셋 구조 및 통계 분석"""
    
    root_dir = Path(root_dir)
    classes = ['crowd', 'fight', 'fall']
    
    print("="*80)
    print("📊 AIHUB CCTV 이상행동 데이터셋 분석")
    print("="*80)
    
    # 통계 저장
    stats = defaultdict(dict)
    
    # Train/Val 비디오 분석
    for split in ['train', 'val']:
        split_path = root_dir / split
        
        if not split_path.exists():
            print(f"\n⚠️  {split} 폴더가 존재하지 않습니다: {split_path}")
            continue
        
        print(f"\n📹 {split.upper()} 비디오:")
        print("-"*80)
        
        total = 0
        
        for class_name in classes:
            class_path = split_path / class_name
            
            if not class_path.exists():
                print(f"  ⚠️  {class_name} 폴더 없음")
                stats[split][class_name] = {'count': 0, 'files': []}
                continue
            
            # 비디오 파일 찾기
            video_files = (
                list(class_path.glob('*.mp4')) +
                list(class_path.glob('*.avi')) +
                list(class_path.glob('*.MP4')) +
                list(class_path.glob('*.AVI'))
            )
            
            count = len(video_files)
            total += count
            
            stats[split][class_name] = {
                'count': count,
                'files': video_files
            }
            
            print(f"  • {class_name:10s}: {count:4d}개")
        
        stats[split]['total'] = total
        print(f"  {'─'*20}")
        print(f"  합계: {total}개")
    
    # JSON 라벨 분석
    print(f"\n📄 JSON 라벨 파일:")
    print("-"*80)
    
    labels_path = root_dir / 'labels_json'
    
    if not labels_path.exists():
        print("  ⚠️  labels_json 폴더가 존재하지 않습니다")
        print("  → JSON 없이 전체 비디오로 학습됩니다")
    else:
        json_count = 0
        for class_name in classes:
            class_path = labels_path / class_name
            
            if not class_path.exists():
                print(f"  ⚠️  {class_name} 폴더 없음")
                continue
            
            json_files = list(class_path.glob('*.json'))
            json_count += len(json_files)
            print(f"  • {class_name:10s}: {len(json_files):4d}개")
        
        print(f"  {'─'*20}")
        print(f"  합계: {json_count}개")
        
        # JSON 매칭 확인
        train_total = stats.get('train', {}).get('total', 0)
        if json_count > 0 and train_total > 0:
            match_rate = (json_count / train_total) * 100
            print(f"\n  📋 JSON 매칭률: {match_rate:.1f}%")
            if match_rate >= 95:
                print(f"  ✅ JSON 라벨 준비 완료!")
            elif match_rate >= 50:
                print(f"  ⚠️  일부 JSON 누락. 누락된 영상은 전체 사용됩니다")
            else:
                print(f"  ❌ JSON 라벨 부족. --use_json False 권장")
    
    # 샘플 비디오 상세 정보
    print(f"\n🎥 샘플 비디오 정보:")
    print("-"*80)
    
    sample_found = False
    for split in ['train', 'val']:
        if split not in stats:
            continue
        
        for class_name in classes:
            if class_name in stats[split]:
                files = stats[split][class_name].get('files', [])
                if files:
                    video_path = files[0]
                    
                    try:
                        cap = cv2.VideoCapture(str(video_path))
                        
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = frame_count / fps if fps > 0 else 0
                            
                            print(f"  파일명: {video_path.name}")
                            print(f"  클래스: {class_name} ({split})")
                            print(f"  해상도: {width} x {height}")
                            print(f"  FPS: {fps:.2f}")
                            print(f"  프레임 수: {frame_count}")
                            print(f"  길이: {duration:.2f}초")
                            
                            cap.release()
                            sample_found = True
                            break
                    
                    except Exception as e:
                        pass
        
        if sample_found:
            break
    
    if not sample_found:
        print("  ⚠️  비디오를 찾을 수 없습니다")
    
    # 샘플 JSON 구조
    if labels_path.exists():
        print(f"\n📋 샘플 JSON 구조:")
        print("-"*80)
        
        sample_found = False
        for class_name in classes:
            class_path = labels_path / class_name
            
            if not class_path.exists():
                continue
            
            json_files = list(class_path.glob('*.json'))
            
            if json_files:
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    print(f"  파일명: {json_files[0].name}")
                    print(f"  클래스: {class_name}")
                    print(f"  최상위 키: {list(data.keys())}")
                    
                    # 이벤트 구간 정보
                    if 'annotations' in data:
                        annotations = data['annotations']
                        if 'event_frame' in annotations:
                            event_frame = annotations['event_frame']
                            print(f"  이벤트 구간: {event_frame}")
                            if event_frame:
                                start, end = event_frame[0]
                                duration_frames = end - start
                                print(f"  이벤트 길이: {duration_frames} 프레임")
                    
                    print(f"\n  구조 미리보기:")
                    json_str = json.dumps(data, ensure_ascii=False, indent=2)
                    print(json_str[:500] + "...")
                    
                    sample_found = True
                    break
                    
                except Exception as e:
                    print(f"  ⚠️  JSON 읽기 오류: {e}")
                    continue
        
        if not sample_found:
            print("  ⚠️  JSON 파일을 찾을 수 없습니다")
    
    # 요약 통계
    print(f"\n📈 요약 통계:")
    print("="*80)
    
    train_total = stats.get('train', {}).get('total', 0)
    val_total = stats.get('val', {}).get('total', 0)
    
    print(f"  총 Train 영상: {train_total}개")
    print(f"  총 Val 영상: {val_total}개")
    print(f"  전체 영상: {train_total + val_total}개")
    print(f"  클래스 수: {len(classes)}개")
    
    # Train 클래스 분포 시각화
    if train_total > 0:
        print(f"\n  📊 Train 클래스 분포:")
        for class_name in classes:
            count = stats.get('train', {}).get(class_name, {}).get('count', 0)
            ratio = count / train_total * 100 if train_total > 0 else 0
            bar_length = int(ratio / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"    {class_name:10s} {bar} {ratio:5.1f}% ({count}개)")
    
    # 클래스 불균형 경고
    if train_total > 0:
        counts = [stats.get('train', {}).get(c, {}).get('count', 0) 
                 for c in classes]
        max_count = max(counts) if counts else 0
        min_count = min([c for c in counts if c > 0], default=0)
        
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 2:
                print(f"\n  ⚠️  클래스 불균형 감지: 최대/최소 = {imbalance_ratio:.1f}배")
                print(f"      → 학습 시 가중치 조정 고려")
    
    print("="*80)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='데이터셋 분석')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='데이터셋 경로 (기본: datasets)')
    
    args = parser.parse_args()
    
    # 경로 확인
    if not Path(args.data_dir).exists():
        print(f"❌ 오류: {args.data_dir} 폴더를 찾을 수 없습니다")
        print(f"📍 현재 경로: {Path.cwd()}")
        print(f"\n💡 해결 방법:")
        print(f"  1. python setup_project.py 실행")
        print(f"  2. datasets/ 폴더에 데이터 배치")
        print(f"  3. 프로젝트 루트에서 실행")
        return
    
    # 분석 실행
    stats = analyze_dataset(args.data_dir)
    
    # 다음 단계 안내
    print("\n💡 다음 단계:")
    print("  python scripts/train.py --model resnet18 --epochs 30 --use_json True")


if __name__ == "__main__":
    main()