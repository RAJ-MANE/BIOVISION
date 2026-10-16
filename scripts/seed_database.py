"""
Utility script to initialize and seed the BioVision SQLite database.
"""
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.repository import get_db_repository

def main():
    print("[Seed] Initializing SQLite database...")
    repo = get_db_repository()
    repo.seed_initial_demo_data()
    batches = repo.get_batches()
    actions = repo.get_actions()
    print(f"[Seed] Successfully seeded {len(batches)} produce lots and {len(actions)} store actions into biovision.db")

if __name__ == "__main__":
    main()
