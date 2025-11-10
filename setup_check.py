"""
Setup Verification Script
Run this to check if your environment is correctly configured
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor} is too old. Need 3.10+")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'langchain',
        'google.generativeai',
        'pinecone',
        'PyPDF2',
        'docx2txt',
        'pandas',
        'python-dotenv'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_installed = False
    
    return all_installed

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Create one using: cp env.example .env")
        return False
    
    print("✅ .env file exists")
    
    # Check for required variables (PINECONE_ENV not needed for serverless)
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = []
    
    with open(env_path) as f:
        content = f.read()
        for var in required_vars:
            if var not in content or f'{var}=your_' in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Please set these variables in .env: {', '.join(missing_vars)}")
        return False
    
    print("✅ Required environment variables are set")
    return True

def check_project_structure():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'ingest.py',
        'query_engine.py',
        'requirements.txt',
        'env.example'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("=" * 50)
    print("RAG Document Assistant - Setup Verification")
    print("=" * 50)
    
    print("\n📋 Checking Python Version...")
    python_ok = check_python_version()
    
    print("\n📦 Checking Dependencies...")
    deps_ok = check_dependencies()
    
    print("\n🔧 Checking Environment Configuration...")
    env_ok = check_env_file()
    
    print("\n📁 Checking Project Structure...")
    structure_ok = check_project_structure()
    
    print("\n" + "=" * 50)
    
    if python_ok and deps_ok and env_ok and structure_ok:
        print("✅ All checks passed! You're ready to run the app.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        if not deps_ok:
            print("\nTo install dependencies:")
            print("   pip install -r requirements.txt")
        if not env_ok:
            print("\nTo configure environment:")
            print("   cp env.example .env")
            print("   # Then edit .env with your API keys")
    
    print("=" * 50)

if __name__ == "__main__":
    main()

