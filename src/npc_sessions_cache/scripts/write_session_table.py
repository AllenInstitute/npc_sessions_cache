from __future__ import annotations

import dataclasses

import npc_lims
import npc_session
import npc_sessions

@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Record:
    """A row in the sessions table"""
    
    # required - should be available for all sessions ------------------ #
    project: str
    session_id: npc_session.SessionRecord
    date: npc_session.DateRecord
    time: npc_session.TimeRecord
    subject: npc_session.SubjectRecord
    subject_age: int
    subject_sex: str
    subject_genotype: str
    rig: str
    experimenters: list[str] | None
    notes: str | None
    issues: list[str]
    intervals: list[str]

    allen_path: str
    cloud_path: str | None
    
    task_version: str | None
    ephys_day: int | None
    behavior_day: int | None
    
    is_ephys: bool
    is_sync: bool
    is_video: bool
    is_templeton: bool
    is_annotated: bool
    is_hab: bool
    is_task: bool
    is_spontaneous: bool
    is_spontaneous_rewards: bool
    is_rf_mapping: bool
    is_optotagging: bool
    is_optotagging_control: bool
    is_opto_perturbation: bool
    is_opto_perturbation_control: bool
    is_injection_perturbation: bool
    
    is_timing_issues: bool
    is_invalid_times: bool
    is_production: bool
    
    is_naive: bool
    is_context_naive: bool # better would be `days_of_context_training`
    
    # currently not possible ------------------------------------------- #
    # is_injection_perturbation_control: bool #! injection metadata not in cloud, Vayle needs to update 
    # is_duragel: bool | None = None
    # virus_name: str | None = None
    # virus_area: str | None = None
    
    # issues dependant - may be none if the session has issues --------- #
    probe_letters_available: str | None = None
    perturbation_areas: list[str] | None = None
    areas_hit: list[str] | None = None
    
    # behavior stuff --------------------------------------------------- #
    n_passing_blocks: int | None = None
    task_duration: float | None = None
    intramodal_dprime_vis: float | None = None
    intramodal_dprime_aud: float | None = None
    intermodal_dprime_vis_blocks: list[float] | None = None
    intermodal_dprime_aud_blocks: list[float] | None = None
    
def get_session_record(session_id: str | npc_session.SessionRecord, session: npc_sessions.DynamicRoutingSession | None = None) -> Record:
    if session is None:
        session = npc_sessions.Session(session_id)
    assert session is not None
    def is_in_intervals(name, intervals):
        return any(name.strip('_').lower() == interval.lower() for interval in intervals)
    return Record(
        project="DynamicRouting" if not session.is_templeton else "Templeton",
        session_id=session.id,
        date=session.id.date,
        time=npc_session.TimeRecord(session.session_start_time.time()),
        subject=session.id.subject,
        subject_age=session.subject.age,
        subject_sex=session.subject.sex,
        subject_genotype=session.subject.genotype,
        rig=session.rig,
        experimenters=session.experimenter,
        notes=session.notes,
        issues=session.info.issues,
        intervals=(intervals := session.epochs[:].stim_name.to_list()),
        allen_path=session.info.allen_path,
        cloud_path=session.info.cloud_path,
        task_version=session.task_version if session.is_task else None,
        ephys_day=session.info.experiment_day if session.is_ephys else None,
        behavior_day=session.info.experiment_day if session.is_task else None,
        is_ephys=session.is_ephys,
        is_sync=session.is_sync,
        is_video=session.is_video,
        is_templeton=session.is_templeton,
        is_annotated=session.is_annotated,
        is_hab=session.is_hab,
        is_task=session.is_task,
        is_spontaneous=is_in_intervals('Spontaneous', intervals),
        is_spontaneous_rewards=is_in_intervals('SpontaneousRewards', intervals),
        is_rf_mapping=is_in_intervals('RFMapping', intervals),
        is_optotagging="optotagging" in session.keywords,
        is_optotagging_control="optotagging_control" in session.keywords,
        is_opto_perturbation=(is_opto := "opto_perturbation" in session.keywords),
        is_opto_perturbation_control="opto_perturbation_control" in session.keywords,
        is_injection_perturbation=session.info.session_kwargs.get('is_injection_perturbation', False), 
        # is_injection_perturbation_control=session.info.session_kwargs.get('is_injection_perturbation_control', False), 
        is_timing_issues="timing_issues" in session.epochs[:].tags.explode().unique(),
        is_invalid_times="invalid_times" in session.epochs[:].tags.explode().unique(),
        is_production=session.info.session_kwargs.get('is_production', True), 
        is_naive=session.info.session_kwargs.get('is_naive', False),
        is_context_naive=session.info.session_kwargs.get('is_context_naive', False) or session.is_templeton,
        probe_letters_available=session.probe_letters_to_use,
        perturbation_areas=sorted((trials := session.trials[:]).opto_labels.unique()) if is_opto else None,
        areas_hit=sorted(session.units[:].structure.unique()) if session.is_annotated else None,
        n_passing_blocks=len((performance := session.performance[:]).query("cross_modal_dprime >= 1.5")) if session.is_task else None,
        task_duration=trials.stop_time.max() - trials.start_time.min() if session.is_task else None,
        intramodal_dprime_vis=performance.vis_intra_dprime.mean() if session.is_task else None,
        intramodal_dprime_aud=performance.aud_intra_dprime.mean() if session.is_task else None,
        intermodal_dprime_vis_blocks=performance.query("rewarded_modality == 'vis'").cross_modal_dprime.to_list() if session.is_task else None,
        intermodal_dprime_aud_blocks=performance.query("rewarded_modality == 'aud'").cross_modal_dprime.to_list() if session.is_task else None,
    )
    
x = get_session_record('DRpilot_668755_20230829')
print(x)