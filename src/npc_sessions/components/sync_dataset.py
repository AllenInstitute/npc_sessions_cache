"""
Dataset object for loading and unpacking an HDF5 dataset generated by
    sync.py

@author: derricw

Allen Institute for Brain Science
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

if TYPE_CHECKING:
    import matplotlib.figure as fig
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

dset_version = 1.04


def get_bit(uint_array: npt.NDArray, bit: int) -> npt.NDArray[np.uint8]:
    """
    Returns a bool array for a specific bit in a uint ndarray.

    Parameters
    ----------
    uint_array : (numpy.ndarray)
        The array to extract bits from.
    bit : (int)
        The bit to extract.

    """
    return np.bitwise_and(uint_array, 2**bit).astype(bool).astype(np.uint8)


class SyncDataset:
    """
    A sync dataset.  Contains methods for loading
        and parsing the binary data.

    Parameters
    ----------
    path : str
        Path to HDF5 file.

    Examples
    --------
    >>> dset = SyncDataset('my_h5_file.h5') # doctest: +SKIP
    >>> logger.info(dset.meta_data) # doctest: +SKIP
    >>> dset.stats() # doctest: +SKIP
    >>> dset.close() # doctest: +SKIP

    >>> with SyncDataset('my_h5_file.h5') as d: # doctest: +SKIP
    ...     logger.info(dset.meta_data)
    ...     dset.stats()

    The sync file documentation from MPE can be found at
    sharepoint > Instrumentation > Shared Documents > Sync_line_labels_discussion_2020-01-27-.xlsx  # NOQA E501
    Direct link:
    https://alleninstitute.sharepoint.com/:x:/s/Instrumentation/ES2bi1xJ3E9NupX-zQeXTlYBS2mVVySycfbCQhsD_jPMUw?e=Z9jCwH


    """

    FRAME_KEYS = ("frames", "stim_vsync", "vsync_stim")
    PHOTODIODE_KEYS = ("photodiode", "stim_photodiode")
    OPTOGENETIC_STIMULATION_KEYS = ("LED_sync", "opto_trial")
    EYE_TRACKING_KEYS = (
        "eye_frame_received",  # Expected eye tracking
        # line label after 3/27/2020
        # clocks eye tracking frame pulses (port 0, line 9)
        "cam2_exposure",
        # previous line label for eye tracking
        # (prior to ~ Oct. 2018)
        "eyetracking",
        "eye_cam_exposing",
        "eye_cam_exposure",
        "eye_cam_json",
        "eye_tracking",
    )  # An undocumented, but possible eye tracking line label  # E114
    BEHAVIOR_TRACKING_KEYS = (
        "beh_frame_received",  # Expected behavior line label after 3/27/2020  # NOQA E127
        # clocks behavior tracking frame # E127
        # pulses (port 0, line 8)
        "cam1_exposure",
        "behavior_monitoring",
    )

    DEPRECATED_KEYS: set[str] = set()

    def __init__(self, path) -> None:
        self.dfile = self.load(path)
        self._check_line_labels()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dfile.filename})"

    def _check_line_labels(self) -> None:
        if hasattr(self, "line_labels"):
            deprecated_keys = set(self.line_labels) & self.DEPRECATED_KEYS
            if deprecated_keys:
                warnings.warn(
                    (
                        f"The loaded sync file contains the "
                        f"following deprecated line label keys: "
                        f"{deprecated_keys}. Consider updating the "
                        f"sync file line labels."
                    ),
                    stacklevel=2,
                )
        else:
            warnings.warn(
                ("The loaded sync file has no line labels and may " "not be valid."),
                stacklevel=2,
            )

    def _process_times(self) -> npt.NDArray[np.int64]:
        """
        Preprocesses the time array to account for rollovers.
            This is only relevant for event-based sampling.

        """
        times = self.get_all_events()[:, 0:1].astype(np.int64)

        intervals = np.ediff1d(times, to_begin=0)
        rollovers = np.where(intervals < 0)[0]

        for i in rollovers:
            times[i:] += 4294967296

        return times

    def load(self, path) -> h5py.File:
        """
        Loads an hdf5 sync dataset.

        Parameters
        ----------
        path : str
            Path to hdf5 file.

        """
        self.dfile = h5py.File(
            path, "r"
        )  # MG edit 3/15 removed 'r' because some sync files were unable to load  # NOQA E501
        self.meta_data: dict[str, Any] = eval(self.dfile["meta"][()])
        self.line_labels: Sequence[str] = self.meta_data["line_labels"]
        self.times = self._process_times()
        return self.dfile

    @property
    def sample_freq(self) -> float:
        try:
            return float(self.meta_data["ni_daq"]["sample_freq"])
        except KeyError:
            return float(self.meta_data["ni_daq"]["counter_output_freq"])

    def get_bit(self, bit: int) -> npt.NDArray[np.uint8]:
        """
        Returns the values for a specific bit.

        Parameters
        ----------
        bit : int
            Bit to return.
        """
        return get_bit(self.get_all_bits(), bit)

    def get_line(self, line: str | int) -> npt.NDArray[np.uint8]:
        """
        Returns the values for a specific line.

        Parameters
        ----------
        line : str
            Line to return.

        """
        bit = self._line_to_bit(line)
        return self.get_bit(bit)

    def get_bit_changes(self, bit: int) -> npt.NDArray[np.uint8]:
        """
        Returns the first derivative of a specific bit.
            Data points are 1 on rising edges and 255 on falling edges.

        Parameters
        ----------
        bit : int
            Bit for which to return changes.

        """
        bit_array = self.get_bit(bit)
        return np.ediff1d(bit_array, to_begin=0)

    def get_line_changes(self, line: str | int) -> npt.NDArray[np.uint8]:
        """
        Returns the first derivative of a specific line.
            Data points are 1 on rising edges and 255 on falling edges.

        Parameters
        ----------
        line : (str)
            Line name for which to return changes.

        """
        bit = self._line_to_bit(line)
        return self.get_bit_changes(bit)

    def get_all_bits(self) -> npt.NDArray:
        """
        Returns the data for all bits.

        """
        return np.array(self.dfile["data"][()][:, -1])

    def get_all_times(self, units: Literal["samples", "seconds"]) -> npt.NDArray[Any]:
        """
        Returns all counter values.

        Parameters
        ----------
        units : str
            Return times in 'samples' or 'seconds'

        """
        if self.meta_data["ni_daq"]["counter_bits"] == 32:
            times = self.get_all_events()[:, 0]
        else:
            times = self.times
        if units.lower() == "samples":
            return times
        elif units.lower() in ["seconds", "sec", "secs"]:
            freq = self.sample_freq
            return times / freq
        else:
            raise ValueError("Only 'samples' or 'seconds' are valid units.")

    def get_all_events(self) -> npt.NDArray[Any]:
        """
        Returns all counter values and their cooresponding IO state.
        """
        return np.array(self.dfile["data"][()])

    def get_events_by_bit(self, bit: int, units=Literal["seconds", "samples"]):
        """
        Returns all counter values for transitions (both rising and falling)
            for a specific bit.

        Parameters
        ----------
        bit : int
            Bit for which to return events.

        """
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes != 0)]

    def get_events_by_line(
        self, line: str | int, units: Literal["samples", "seconds"] = "samples"
    ):
        """
        Returns all counter values for transitions (both rising and falling)
            for a specific line.

        Parameters
        ----------
        line : str
            Line for which to return events.

        """
        line = self._line_to_bit(line)
        return self.get_events_by_bit(line, units)

    def _line_to_bit(self, line: str | int) -> int:
        """
        Returns the bit for a specified line.  Either line name and number is
            accepted.

        Parameters
        ----------
        line : str
            Line name for which to return corresponding bit.

        """
        if type(line) is int:
            return line
        elif type(line) is str:
            return self.line_labels.index(line)
        else:
            raise TypeError("Incorrect line type.  Try a str or int.")

    def _bit_to_line(self, bit: int) -> str:
        """
        Returns the line name for a specified bit.

        Parameters
        ----------
        bit : int
            Bit for which to return the corresponding line name.
        """
        return self.line_labels[bit]

    def get_rising_edges(
        self, line: str | int, units: Literal["samples", "seconds"] = "samples"
    ) -> npt.NDArray[Any]:
        """
        Returns the counter values for the rizing edges for a specific bit or
            line.

        Parameters
        ----------
        line : str
            Line for which to return edges.

        """
        bit = self._line_to_bit(line)
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes == 1)]

    def get_edges(
        self,
        kind: Literal["rising", "falling", "all"],
        keys: str | Sequence[str],
        units: Literal["seconds", "samples"],
        permissive: bool = False,
    ) -> npt.NDArray | None:
        """Utility function for extracting edge times from a line

        Parameters
        ----------
        kind : One of "rising", "falling", or "all". Should this method return
            timestamps for rising, falling or both edges on the appropriate
            line
        keys : These will be checked in sequence. Timestamps will be returned
            for the first which is present in the line labels
        units : one of "seconds", "samples", or "indices". The returned
            "time"stamps will be given in these units.
        raise_missing : If True and no matching line is found, a KeyError will
            be raised

        Returns
        -------
        An array of edge times. If raise_missing is False and none of the keys
            were found, returns None.

        Raises
        ------
        KeyError : none of the provided keys were found among this dataset's
            line labels

        """
        if kind == "falling":
            fn = self.get_falling_edges
        elif kind == "rising":
            fn = self.get_rising_edges
        elif kind == "all":
            rising = (self.get_edges("rising", keys, units),)
            falling = self.get_edges("falling", keys, units)
            assert rising and falling
            return np.sort(np.concatenate([rising, falling]))

        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            try:
                result = fn(key, units)
            except ValueError:
                continue
            else:
                return result

        if not permissive:
            raise KeyError(f"none of {keys} were found in this dataset's line labels")
        return None

    def get_falling_edges(
        self, line: str | int, units: Literal["samples", "seconds"] = "samples"
    ):
        """
        Returns the counter values for the falling edges for a specific bit
            or line.

        Parameters
        ----------
        line : str
            Line for which to return edges.

        """
        bit = self._line_to_bit(line)
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes == 255)]

    def get_nearest(
        self,
        source: str,
        target: str,
        source_edge: Literal["rising", "falling"] = "rising",
        target_edge: Literal["rising", "falling"] = "rising",
        direction: Literal["previous", "next"] = "previous",
        units: Literal["indices", "samples", "seconds"] = "indices",
    ) -> npt.NDArray:
        """
        For all values of the source line, finds the nearest edge from the
            target line.

        By default, returns the indices of the target edges.

        Args:
            source (str, int): desired source line
            target (str, int): desired target line
            source_edge [Optional(str)]: "rising" or "falling" source edges
            target_edge [Optional(str): "rising" or "falling" target edges
            direction (str): "previous" or "next". Whether to prefer the
                previous edge or the following edge.
            units (str): "indices"

        """
        source_edges = getattr(self, f"get_{source_edge.lower()}_edges")(
            source.lower(), units="samples"
        )  # E501
        target_edges = getattr(self, f"get_{target_edge.lower()}_edges")(
            target.lower(), units="samples"
        )  # E501
        indices = np.searchsorted(target_edges, source_edges, side="right")
        if direction.lower() == "previous":
            indices[np.where(indices != 0)] -= 1
        elif direction.lower() == "next":
            indices[np.where(indices == len(target_edges))] = -1
        if units in ["indices", "index"]:
            return indices
        elif units == "samples":
            return target_edges[indices]
        elif units in ["sec", "seconds", "second"]:
            return target_edges[indices] / self.sample_freq
        else:
            raise KeyError("Invalid units.  Try 'seconds', 'samples' or 'indices'")

    def get_analog_channel(
        self,
        channel: int,
        start_time: float = 0.0,
        stop_time: float | None = None,
        downsample: int = 1,
    ) -> npt.NDArray:
        """
        Returns the data from the specified analog channel between the
            timepoints.

        Args:
            channel (int, str): desired channel index or label
            start_time (Optional[float]): start time in seconds
            stop_time (Optional[float]): stop time in seconds
            downsample (Optional[int]): downsample factor

        Returns:
            ndarray: slice of data for specified channel

        Raises:
            KeyError: no analog data present

        """
        if isinstance(channel, str):
            channel_index = self.analog_meta_data["analog_labels"].index(channel)
            channel = self.analog_meta_data["analog_channels"].index(channel_index)

        if "analog_data" in self.dfile.keys():
            dset = np.array(self.dfile["analog_data"])
            analog_meta = self.get_analog_meta()
            sample_rate = analog_meta["analog_sample_rate"]
            start = int(start_time * sample_rate)
            if stop_time:
                stop = int(stop_time * sample_rate)
                return dset[start:stop:downsample, channel]
            else:
                return dset[start::downsample, channel]
        else:
            raise KeyError("No analog data was saved.")

    def get_analog_meta(self) -> Any:
        """
        Returns the metadata for the analog data.
        """
        if "analog_meta" in self.dfile.keys():
            return eval(self.dfile["analog_meta"].value)
        else:
            raise KeyError("No analog data was saved.")

    @property
    def analog_meta_data(self) -> Any:
        return self.get_analog_meta()

    def line_stats(self, line, print_results=True) -> dict[str, Any] | None:
        """
        Quick-and-dirty analysis of a bit.

        ##TODO: Split this up into smaller functions.

        """
        # convert to bit
        bit = self._line_to_bit(line)

        # get the bit's data
        bit_data = self.get_bit(bit)
        total_data_points = len(bit_data)

        # get the events
        events = self.get_events_by_bit(bit)
        total_events = len(events)

        # get the rising edges
        rising = self.get_rising_edges(bit)
        total_rising = len(rising)

        # get falling edges
        falling = self.get_falling_edges(bit)
        total_falling = len(falling)

        if total_events <= 0:
            if print_results:
                logger.info("*" * 70)
                logger.info("No events on line: %s" % line)
                logger.info("*" * 70)
            return None
        elif total_events <= 10:
            if print_results:
                logger.info("*" * 70)
                logger.info("Sparse events on line: %s" % line)
                logger.info("Rising: %s" % total_rising)
                logger.info("Falling: %s" % total_falling)
                logger.info("*" * 70)
            return {
                "line": line,
                "bit": bit,
                "total_rising": total_rising,
                "total_falling": total_falling,
                "avg_freq": None,
                "duty_cycle": None,
            }
        else:
            # period
            period = self.period(line)

            avg_period = period["avg"]
            max_period = period["max"]
            min_period = period["min"]
            period_sd = period["sd"]

            # freq
            avg_freq = self.frequency(line)

            # duty cycle
            duty_cycle = self.duty_cycle(line)

            if print_results:
                logger.info("*" * 70)

                logger.info("Quick stats for line: %s" % line)
                logger.info("Bit: %i" % bit)
                logger.info("Data points: %i" % total_data_points)
                logger.info("Total transitions: %i" % total_events)
                logger.info("Rising edges: %i" % total_rising)
                logger.info("Falling edges: %i" % total_falling)
                logger.info("Average period: %s" % avg_period)
                logger.info("Minimum period: %s" % min_period)
                logger.info("Max period: %s" % max_period)
                logger.info("Period SD: %s" % period_sd)
                logger.info("Average freq: %s" % avg_freq)
                logger.info("Duty cycle: %s" % duty_cycle)

                logger.info("*" * 70)

            return {
                "line": line,
                "bit": bit,
                "total_data_points": total_data_points,
                "total_events": total_events,
                "total_rising": total_rising,
                "total_falling": total_falling,
                "avg_period": avg_period,
                "min_period": min_period,
                "max_period": max_period,
                "period_sd": period_sd,
                "avg_freq": avg_freq,
                "duty_cycle": duty_cycle,
            }

    def period(
        self, line: str | int, edge: Literal["rising", "falling"] = "rising"
    ) -> dict[str, Any]:
        """
        Returns a dictionary with avg, min, max, and st of period for a line.
        """
        bit = self._line_to_bit(line)

        if edge.lower() == "rising":
            edges = self.get_rising_edges(bit)
        elif edge.lower() == "falling":
            edges = self.get_falling_edges(bit)
        else:
            raise ValueError(
                f'edge should be one of ("rising", "falling") not {edge!r}'
            )

        if len(edges) > 2:
            timebase_freq = self.meta_data["ni_daq"]["counter_output_freq"]
            avg_period = np.mean(np.ediff1d(edges[1:])) / timebase_freq
            max_period = np.max(np.ediff1d(edges[1:])) / timebase_freq
            min_period = np.min(np.ediff1d(edges[1:])) / timebase_freq
            period_sd = np.std(avg_period)

        else:
            raise IndexError("Not enough edges for period: %i" % len(edges))

        return {
            "avg": avg_period,
            "max": max_period,
            "min": min_period,
            "sd": period_sd,
        }

    def frequency(
        self, line: str | int, edge: Literal["rising", "falling"] = "rising"
    ) -> float:
        """
        Returns the average frequency of a line.
        """

        period = self.period(line, edge)
        return 1.0 / period["avg"]

    def duty_cycle(self, line: str | int) -> Literal["fix me"]:
        """
        Doesn't work right now.  Freezes python for some reason.

        Returns the duty cycle of a line.

        """
        return "fix me"
        bit = self._line_to_bit(line)

        rising = self.get_rising_edges(bit)
        falling = self.get_falling_edges(bit)

        total_rising = len(rising)
        total_falling = len(falling)

        if total_rising > total_falling:
            rising = rising[:total_falling]
        elif total_rising < total_falling:
            falling = falling[:total_rising]
        else:
            pass

        if rising[0] < falling[0]:
            # line starts low
            high = falling - rising
        else:
            # line starts high
            high = np.concatenate(
                falling, self.get_all_events()[-1, 0]
            ) - np.concatenate(0, rising)

        total_high_time = np.sum(high)
        all_events = self.get_events_by_bit(bit)
        total_time = all_events[-1] - all_events[0]
        return 1.0 * total_high_time / total_time

    def stats(self) -> list[dict[str, Any]]:
        """
        Quick-and-dirty analysis of all bits.  Prints a few things about each
            bit where events are found.
        """
        bits = []
        for i in range(32):
            bits.append(self.line_stats(i, print_results=False))
        active_bits = [x for x in bits if x is not None]
        logger.info("Active bits: ", len(active_bits))
        for bit in active_bits:
            logger.info("*" * 70)
            logger.info("Bit: %i" % bit["bit"])
            logger.info("Label: %s" % self.line_labels[bit["bit"]])
            logger.info("Rising edges: %i" % bit["total_rising"])
            logger.info("Falling edges: %i" % bit["total_falling"])
            logger.info("Average freq: %s" % bit["avg_freq"])
            logger.info("Duty cycle: %s" % bit["duty_cycle"])
        logger.info("*" * 70)
        return active_bits

    def plot_all(
        self,
        start_time: float,
        stop_time: float | None = None,
        auto_show: bool = True,
    ) -> None:
        """
        Plot all active bits.

        Yikes.  Come up with a better way to show this.

        """
        import matplotlib.pyplot as plt

        for bit in range(32):
            if len(self.get_events_by_bit(bit)) > 0:
                self.plot_bit(
                    bit,
                    start_time,
                    stop_time,
                    auto_show=False,
                )
        if auto_show:
            plt.show()

    def plot_bits(
        self,
        bits: Sequence[int],
        start_time: float = 0.0,
        end_time: float | None = None,
        auto_show: bool = True,
    ) -> tuple[fig.Figure, plt.Axes | Iterable[plt.Axes]]:
        """
        Plots a list of bits.
        """
        import matplotlib.pyplot as plt

        subplots = len(bits)
        f, axes = plt.subplots(subplots, sharex=True, sharey=True)
        if not isinstance(axes, Iterable):
            axes = [axes]

        for bit, ax in zip(bits, axes):
            self.plot_bit(bit, start_time, end_time, auto_show=False, axes=ax)
        # f.set_size_inches(18, 10, forward=True)
        f.subplots_adjust(hspace=0)

        if auto_show:
            plt.show()

        return f, axes

    def plot_bit(
        self,
        bit,
        start_time: float = 0.0,
        end_time: float | None = None,
        auto_show: bool = True,
        axes=None,
        name="",
    ) -> tuple[fig.Figure, plt.Axes | Iterable[plt.Axes]]:
        """
        Plots a specific bit at a specific time period.
        """
        import matplotlib.pyplot as plt

        times = self.get_all_times(units="seconds")
        if not end_time:
            end_time = 2**32

        window = (times < end_time) & (times > start_time)

        if axes:
            ax = axes
        else:
            ax = plt

        if not name:
            name = self._bit_to_line(bit)
        if not name:
            name = str(bit)

        bit = self.get_bit(bit)
        ax.step(times[window], bit[window], where="post")
        if hasattr(ax, "set_ylim"):
            ax.set_ylim(-0.1, 1.1)
        else:
            axes_obj = plt.gca()
            axes_obj.set_ylim(-0.1, 1.1)
        # ax.set_ylabel('Logic State')
        # ax.yaxis.set_ticks_position('none')
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel("time (seconds)")
        ax.legend([name])

        if auto_show:
            plt.show()

        return plt.gcf()

    def plot_line(
        self,
        line,
        start_time: float = 0.0,
        end_time: float | None = None,
        auto_show: bool = True,
    ) -> None:
        """
        Plots a specific line at a specific time period.
        """
        import matplotlib.pyplot as plt

        bit = self._line_to_bit(line)
        self.plot_bit(bit, start_time, end_time, auto_show=False)

        # plt.legend([line])
        if auto_show:
            plt.show()

    def plot_lines(
        self,
        lines,
        start_time: float = 0.0,
        end_time: float | None = None,
        auto_show: bool = True,
    ) -> tuple[fig.Figure, plt.Axes | Iterable[plt.Axes]]:
        """
        Plots specific lines at a specific time period.
        """
        import matplotlib.pyplot as plt

        bits = []
        for line in lines:
            bits.append(self._line_to_bit(line))
        f, axes = self.plot_bits(
            bits,
            start_time,
            end_time,
            auto_show=False,
        )

        plt.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.95)
        if auto_show:
            plt.show()

        return f, axes

    def close(self) -> None:
        """
        Closes the dataset.
        """
        self.dfile.close()

    def __enter__(self) -> Self:
        """
        So we can use context manager (with...as) like any other open file.

        Examples
        --------
        >>> with Dataset('my_data.h5') as d: # doctest: +SKIP
        ...     d.stats()

        """
        return self

    def __exit__(self, type, value, traceback) -> None:
        """
        Exit statement for context manager.
        """
        self.close()


if __name__ == "__main__":
    pass
