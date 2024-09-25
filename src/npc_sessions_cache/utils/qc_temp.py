"""QC temp file for testing purposes. To be refactored and wrapped into
 session_qc probably...
"""
from __future__ import annotations

import upath
from aind_data_schema.core import quality_control
from aind_data_schema_models.modalities import Modality


def _parse_dynamic_routing_asset_path_meta_info(
    path: upath.UPath,
) -> tuple[str, str]:
    """Extracts asset record from a dynamic routing asset path.

    >>> parsed = _parse_dynamic_routing_asset_path_meta_info(upath.UPath("timing/assorted_lick_times/620263_2022-07-26_0.png"))
    >>> assert parsed[0] == "assorted_lick_times"
    """
    return path.parent.name, path.parent.parent.name


def paths_to_QualityControl(
    paths: list[upath.UPath],
) -> quality_control.QualityControl:
    """
    >>> paths = [upath.UPath('/timing/assorted_lick_times/620263_2022-07-26_0.png'), upath.UPath('/timing/assorted_lick_times/620263_2022-07-26_1.png'), upath.UPath('behavior/running/620263_2022-07-26_0.png')]
    >>> qc = paths_to_QualityControl(paths)
    >>> assert len(qc.evaluations) == 2
    >>> assert qc.evaluations[0].evaluation_name == "timing"
    >>> assert qc.evaluations[0].qc_metrics[0].reference.startswith("/timing/assorted_lick_times")
    """
    qc_metadata = {}
    for path in paths:
        minor_type, major_type = \
            _parse_dynamic_routing_asset_path_meta_info(path)
        if major_type not in qc_metadata:
            qc_metadata[major_type] = {
                minor_type: [path]
            }
        if minor_type not in qc_metadata[major_type]:
            qc_metadata[major_type][minor_type] = [path]
        else:
            qc_metadata[major_type][minor_type].append(path)

    qc_evaluations = []
    for major_type_name, minor_types in qc_metadata.items():
        qc_metrics = []
        for minor_type_name, minor_paths in minor_types.items():
            for idx, path in enumerate(minor_paths):
                qc_metrics.append(
                    quality_control.QCMetric(
                        name=f"{minor_type_name}_{idx}",
                        value=None,
                        reference=path.as_posix(),
                    )
                )
        qc_evaluations.append(
            quality_control.QCEvaluation(
                evaluation_stage=quality_control.Stage.PROCESSING,
                evaluation_modality=Modality.BEHAVIOR,
                evaluation_name=major_type_name,
                qc_metrics=qc_metrics,
            )
        )

    return quality_control.QualityControl(
        evaluations=qc_evaluations,
    )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )