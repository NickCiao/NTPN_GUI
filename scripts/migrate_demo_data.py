#!/usr/bin/env python3
"""
Migration script to convert demo data from unsafe pickle to safe NPZ format.

Usage:
    python scripts/migrate_demo_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import ntpn modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ntpn.data_loaders import convert_to_safe_format, DataLoadError


def main():
    """Migrate demo data files to safe NPZ format."""
    # Define demo data paths
    demo_dir = Path('data/demo_data')

    st_file = demo_dir / 'raw_stbins.p'
    context_file = demo_dir / 'context_labels.p'
    context_key = 'context_labels'  # Note: plural!

    print("=" * 80)
    print("NTPN Demo Data Migration Tool")
    print("=" * 80)
    print()
    print("This script converts unsafe pickle files to safe NPZ format.")
    print()
    print(f"Input files:")
    print(f"  {st_file}")
    print(f"  {context_file}")
    print()

    # Check if files exist
    if not st_file.exists():
        print(f"ERROR: Spike data file not found: {st_file}")
        sys.exit(1)

    if not context_file.exists():
        print(f"ERROR: Context file not found: {context_file}")
        sys.exit(1)

    try:
        # Perform conversion
        output_st, output_context = convert_to_safe_format(
            st_file,
            context_file,
            context_key,
            output_dir=demo_dir
        )

        print()
        print("=" * 80)
        print("SUCCESS! Migration complete.")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. The application will now automatically use the safe NPZ files")
        print("  2. You can delete the old pickle files if desired:")
        print(f"       rm {st_file}")
        print(f"       rm {context_file}")
        print()

    except DataLoadError as e:
        print(f"ERROR: Migration failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
