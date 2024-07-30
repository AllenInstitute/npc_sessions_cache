import datetime
import json
import pathlib

import pandas as pd

BASE_PATH = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/session_metadata")
RECORDS_PATH = BASE_PATH / "records"
TABLES_PATH = BASE_PATH / "tables"

def cleanup_old_tables() -> None:
    
    print("Removing all but the latest xlsx table files...")
    # watchout - new/open files have a hidden extra file starting with ~$
    for path in sorted(TABLES_PATH.glob("sessions*.xlsx"))[:-1]:
        try:
            path.unlink(missing_ok=True)
            print(f"Deleted {path.relative_to(TABLES_PATH).as_posix()}")
        except PermissionError:
            print(f"Couldn't delete {path.relative_to(TABLES_PATH).as_posix()}: likely open elsewhere")
    print("Done")
    
def write_session_table_from_records() -> None:
    
    print("Writing session table from json records...")
    
    records = []
    for idx, generic_path in enumerate(RECORDS_PATH.glob("*.json")):
        records.append(json.loads(generic_path.read_text()))
    
    df = pd.DataFrame.from_records(records)
    print(f"Created table from {idx + 1} session records")
    
    generic_path = TABLES_PATH / "sessions.xlsx"
    generic_path.parent.mkdir(parents=True, exist_ok=True)
    
    # give each excel file a new name, as they can't be overwritten if open in Excel:
    dt = datetime.datetime.now().isoformat(sep="_", timespec="seconds").replace(":", "-")
    path = generic_path.with_stem(f"{generic_path.stem}_{dt}")
    df.to_excel(path, index=False, freeze_panes=(1, 2)) # freeze top row and two leftmost columns
    print(f"Wrote table to {path.relative_to(BASE_PATH).as_posix()}")
    
    # parquet files are more likely to be opened programatically, so having the
    # fixed name with no datetime suffix is preferable:
    path = generic_path.with_suffix(".parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote table to {path.relative_to(BASE_PATH).as_posix()}")
    
    print("Done")

if __name__ == "__main__":
    write_session_table_from_records()
    cleanup_old_tables()