import sys
import os

# 將當前工作目錄加入 Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smc_trader.main import main

if __name__ == "__main__":
    main()
