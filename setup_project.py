"""
프로젝트 디렉토리 구조 자동 생성
"""

from pathlib import Path


def create_project_structure():
    """프로젝트 구조 생성"""
    
    print("="*80)
    print("🚀 CCTV 이상행동 탐지 프로젝트 초기화")
    print("="*80)
    
    # 디렉토리 구조
    directories = [
        # 데이터셋
        "datasets/train/crowd",
        "datasets/train/fight",
        "datasets/train/fall",
        "datasets/val/crowd",
        "datasets/val/fight",
        "datasets/val/fall",
        
        # 소스 코드
        "src/data",
        "src/models",
        "src/utils",
        
        # 스크립트
        "scripts",
        
        # 결과 저장
        "checkpoints",
        "logs",
        "results",
    ]
    
    print("\n📁 디렉토리 생성 중...")
    created = 0
    
    for dir_path in directories:
        full_path = Path(dir_path)
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path}")
            created += 1
        else:
            print(f"  ⏭️  {dir_path} (이미 존재)")
    
    print(f"\n✨ {created}개 디렉토리 생성 완료!")
    
    # __init__.py 파일 생성
    create_init_files()
    
    # .gitignore 생성
    create_gitignore()
    
    print("\n" + "="*80)
    print("🎉 프로젝트 초기화 완료!")
    print("="*80)
    print("\n📋 다음 단계:")
    print("  1. datasets/ 폴더에 AIHUB 데이터 배치")
    print("  2. python -m venv venv")
    print("  3. source venv/bin/activate")
    print("  4. pip install -r requirements.txt")
    print("  5. python scripts/analyze_dataset.py")
    print("="*80)


def create_init_files():
    """__init__.py 파일 생성"""
    
    print("\n📄 __init__.py 파일 생성 중...")
    
    init_files = {
        "src/__init__.py": '"""CCTV 이상행동 탐지 프로젝트"""\n\n__version__ = "1.0.0"\n',
        "src/data/__init__.py": '"""데이터 로딩 모듈"""\n\nfrom .dataset import CCTVDataset\nfrom .dataloader import get_dataloaders\n\n__all__ = ["CCTVDataset", "get_dataloaders"]\n',
        "src/models/__init__.py": '"""모델 정의 모듈"""\n\nfrom .backbones import build_model, MODEL_INFO\n\n__all__ = ["build_model", "MODEL_INFO"]\n',
        "src/utils/__init__.py": '"""유틸리티 모듈"""\n\n__all__ = []\n',
    }
    
    for file_path, content in init_files.items():
        path = Path(file_path)
        if not path.exists():
            path.write_text(content, encoding='utf-8')
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⏭️  {file_path} (이미 존재)")


def create_gitignore():
    """gitignore 파일 생성"""
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        print("\n⏭️  .gitignore 이미 존재")
        return
    
    gitignore_content = """# 데이터셋
datasets/

# 체크포인트
checkpoints/*.pth

# 로그
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Results
results/*.png
results/*.csv
"""
    
    gitignore_path.write_text(gitignore_content, encoding='utf-8')
    print("\n✅ .gitignore 생성 완료")


if __name__ == "__main__":
    create_project_structure()