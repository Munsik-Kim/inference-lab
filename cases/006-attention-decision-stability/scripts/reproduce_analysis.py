"""Recalculate retained results into a new external directory. Never run a model."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.analysis import summarize
from src.storage import external_directory, write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=external_directory(a.output,a.case)
    summary,pairs=summarize(a.case)
    write_new(out/'summary.json',summary);write_new(out/'pairs.json',pairs)
    print(f"Recalculated {summary['measured_record_count']} measured records; {summary['status']['execution_status']}")


if __name__=='__main__':
    main()
