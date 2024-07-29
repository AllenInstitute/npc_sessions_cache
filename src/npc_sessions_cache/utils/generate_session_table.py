import datetime
import json
import pathlib

import pandas as pd

BASE_PATH = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/session_metadata")
RECORDS_PATH = BASE_PATH / "records"
TABLES_PATH = BASE_PATH / "tables"

def cleanup_old_tables() -> None:
    
    print("Removing all but the latest xlsx table files...")
    for path in sorted(TABLES_PATH.glob("*.xlsx"))[:-1]:
        try: 
            path.unlink(missing_ok=True)
        except PermissionError:
            print(f"Couldn't delete {path.relative_to(TABLES_PATH).as_posix()}: likely open elsewhere")
    print("Done")
    
def write_session_table_from_records() -> None:
    
    print("Writing session table from json records...")
    
    records = []
    for idx, path in enumerate(RECORDS_PATH.glob("*.json")):
        records.append(json.loads(path.read_text()))
    
    df = pd.DataFrame.from_records(records)
    print(f"Created table from {idx + 1} session records")
    
    path = TABLES_PATH / "sessions.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # give each excel file a new name, as they can't be overwritten if open in Excel:
    dt = datetime.datetime.now().isoformat(sep="_", timespec="seconds").replace(":", "-")
    df.to_excel(path.with_stem(f"{path.stem}_{dt}"), index=False)
    print(f"Wrote table to {path.relative_to(TABLES_PATH).as_posix()}")
    
    # parquet files are more likely to be opened programatically, so having the
    # easier-to-remember name with no datetime suffix is preferable:
    df.to_parquet(path.with_suffix(".parquet"), index=False)
    print(f"Wrote table to {path.relative_to(TABLES_PATH).as_posix()}")
    
    print("Done")

if __name__ == "__main__":
    write_session_table_from_records()
    cleanup_old_tables()