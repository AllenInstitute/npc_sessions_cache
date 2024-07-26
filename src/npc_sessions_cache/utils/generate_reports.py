"""
Dump html reports for each session, pointing to qc images in the cloned s3 bucket.

Should be runnable from a bat file on the Isilon so python stdlib only.
"""

import concurrent.futures
import functools
import pathlib

QC_PATH = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/qc")
CSS_PATH = next(pathlib.Path(__file__).parent.rglob("single_page_img_json_report.css"))
QC_REPORTS_PATH = pathlib.Path(__file__).parent / "by_session"

DOC = """
<!DOCTYPE html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="description" content="">
        <title>{name}</title>
        <h1 text-align="middle">{name}</h1>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="{css_path}">
        <h1>qc</h1><div class="row"><div class="column">
        </div></div>
    </head>
    <body>
    {content}
    </body>
</html>
"""

MODULE_HEADING_TAG = """
<h1>{name}<div class="row"><div class="column">
{content}
</h1>
"""

FUNCTION_HEADING_TAG = """
<h2>{name}<div class="row"><div class="column">
{content}
</h2>
"""

FIG_TAG = """
<figure>
  <img src="{source}" alt="{source}"></img>
  <figcaption><b><a href="file:///{source}" target="_blank">link</a></b></figcaption>
</figure>
"""

JSON_TAG = """
<figure>
  <iframe src="{source}" title="{source}"></iframe>
  <figcaption><b><a href="file:///{source}" target="_blank">link</a></b></figcaption>
</figure>
"""

COLUMN_TAG = """
</div><div class="column">
"""

TEXT_TAG = """
<p>{content}</p>
"""

ERROR_TAG = """
<p style="color:red;">{content}</p>
"""

class Data:
    __fields__ = ["path"]

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path

    @property
    def session_id(self) -> str:
        return "_".join(self.path.stem.split("_")[:2])

    @property
    def date(self) -> str:
        return self.session_id.split("_")[1]

    @property
    def index(self) -> int:
        try:
            return int(self.path.stem.split("_")[2])
        except IndexError:
            return 0

    @property
    def module(self) -> str:
        return self.path.parent.parent.name

    @property
    def function(self) -> str:
        return self.path.parent.name

    @property
    def is_json(self) -> bool:
        return self.path.suffix == ".json"

    @property
    def is_fig(self) -> bool:
        return self.path.suffix == ".png"

    @property
    def is_text(self) -> bool:
        return self.path.suffix == ".txt"

    @property
    def is_error(self) -> bool:
        return self.path.suffix == ".error"

    @property
    def tag(self) -> str:
        if self.is_fig:
            return FIG_TAG.format(
                source=str(self.path),
                name=self.path.stem
            )
        if self.is_json:
            return JSON_TAG.format(
                source=str(self.path),
                name=self.path.stem,
            )
        if self.is_text:
            return TEXT_TAG.format(
                content=self.path.read_text().replace("\n", "<br>")
            )
        if self.is_error:
            return ERROR_TAG.format(
                content=self.path.read_text().replace("\n", "<br>")
            )
        raise NotImplementedError(f"Unknown file type: {self.path}")

def get_session_html(session_id: str) -> str:
    body = ""
    modules_and_functions: dict[str, dict[str, list[str]]] = {}
    for d in get_session_qc_data(session_id):
        modules_and_functions.setdefault(d.module, {}).setdefault(d.function, []).append(d.tag + COLUMN_TAG)
    for module, functions in modules_and_functions.items():
        body += MODULE_HEADING_TAG.format(
            name=module,
            content="".join(
                FUNCTION_HEADING_TAG.format(
                    name=function,
                    content="".join(tags)
                ) for function, tags in functions.items()
            )
        )
    return DOC.format(
        name=session_id,
        css_path=str(CSS_PATH),
        content=body
    )

def write_session_html(session_id: str) -> None:
    (QC_REPORTS_PATH / f"{session_id}.html").write_text(get_session_html(session_id))

@functools.cache
def get_all_qc_data() -> tuple[Data, ...]:
    return tuple(
        Data(p)
        for p in QC_PATH.rglob('*')
        if not p.is_dir()
        and p.name != "Thumbs.db"
    )

def get_session_qc_data(session_id: str) -> tuple[Data, ...]:
    return tuple(
        sorted(
            (d for d in get_all_qc_data() if d.session_id == session_id and 'utils' not in (d.module, d.function)),
            key=lambda d: d.index)
        )

def write_all_session_htmls() -> None:
    print("Writing session html files...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_session_id = {}
        for session_id in set(d.session_id for d in get_all_qc_data()):
            future = executor.submit(write_session_html, session_id)
            future_to_session_id[future] = session_id
        for future in concurrent.futures.as_completed(future_to_session_id):
            print(f"Wrote {future_to_session_id[future]}\r", end="", flush=True)
    print(f"Finished writing {len(future_to_session_id)} session html files to: {QC_REPORTS_PATH}")
    print(f"Latest session: {max(get_all_qc_data(), key=lambda d: d.date).session_id}")

if __name__ == "__main__":
    write_all_session_htmls()
