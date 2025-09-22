import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import find_peaks
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from irl_scg import config
from irl_scg.krippendorff_alpha import *

MAT_DIR = config.RAW_DATA_DIR / "hypovolemia-annotations"
MAT_FILES = [(MAT_DIR / f) for f in os.listdir(MAT_DIR) if f.endswith(".mat")]
MAT_CATHETER_DIR = config.RAW_DATA_DIR / "hypovolemia-catheter-python" / "ao-extract-2"
FILE_NUMBERS = [
    1.1,
    1.2,
    1.3,
    1.4,
    2.1,
    2.2,
    2.3,
    2.4,
    3.1,
    3.2,
    3.3,
    4.1,
    4.2,
    5.1,
    5.2,
    5.3,
    6.1,
    6.2,
    6.3,
    6.4,
]
FILE_NUMBERS_BY_PIG = [
    [1.1, 1.2, 1.3, 1.4],
    [2.1, 2.2, 2.3, 2.4],
    [3.1, 3.2, 3.3],
    [4.1, 4.2],
    [5.1, 5.2, 5.3],
    [6.1, 6.2, 6.3, 6.4],
]
FS = 2000


class HypoDataset:
    def __init__(self, file_number):
        """
        Initializes the HypoDataset object.
        Args:
            file_number (int or str): The identifier for the file(s) to be loaded.
                                      If an integer is provided, it will be converted to a string.
        Attributes:
            file_name (str): The string representation of the file_number.
            files (list): A list of file paths from MAT_FILES that match the file_name.
            data (list): A list to store the data loaded from the files.
        Methods:
            load_data(file): Loads data from the specified file.
            fix_points(): Processes the loaded data to fix points.
        """
        self.file_number = file_number
        self.file_name = str(file_number)
        if isinstance(file_number, int):
            self.files = [
                f
                for f in MAT_FILES
                if self.file_name in str(os.path.basename(f).split(".")[0])
            ]
        else:
            self.files = [
                f for f in MAT_FILES if self.file_name in str(os.path.basename(f))
            ]
        self.data = []
        for file in self.files:
            self.data.append(self.load_data(file, verbose=False))
        if int(file_number) == 5 or int(file_number) == 2:
            self.ao_annot_points_median = self.return_median_points_all("ao")
            self.ac_annot_points_median = self.return_median_points_all("ac")
        else:
            self.fix_points()
            self.ao_annot_points_median = self.return_median_points("ao")
            self.ac_annot_points_median = self.return_median_points("ac")
            self.acv_annot_points_median = self.return_median_points("acv")
            self.mo_annot_points_median = self.return_median_points("mo")
        self.scg_beats = self.data[0]["scgBeats"]

    def load_data(self, file_name, verbose=False):
        """
        Loads data from a .mat file.
        Parameters:
            file_name (str): The path to the .mat file to be loaded.
        Returns:
            dict: The data contained in the .mat file.
        Raises:
            Exception: If there is an error loading the .mat file, an exception is caught and an error message is printed.
        """

        try:
            if verbose:
                print(f"Data {os.path.basename(file_name)} loaded...")
            # Read the catheter data
            pig_catheter_dir = MAT_CATHETER_DIR / f"pig-{int(self.file_number)}"
            pig_file_path = pig_catheter_dir / f"S{self.file_number}_final.mat"
            if int(self.file_number) == 5 or int(self.file_number) == 2:
                pass
            else:
                mat_data_cath = scipy.io.loadmat(pig_file_path)
                self.ao_cath_points = mat_data_cath["ao_cath"]
                self.ac_cath_points = mat_data_cath["ac_cath"]
                self.ap_beats = mat_data_cath["ap_beats"]
            return scipy.io.loadmat(file_name)
        except Exception as e:
            print(f"Error loading .mat file: {e}")

    def fix_points(self):
        """
        Adjusts the point values in the dataset.
        This method iterates over the dataset and modifies the values of specific point types.
        For 'ACMaxPoints', 'ACvPoints', and 'MOPoints', if the value is greater than 600, it is divided by 4.
        For 'AOMaxPoints', if the value is greater than 150, it is divided by 4.
        Returns:
            None
        """

        point_types = ["ACMaxPoints", "ACvPoints", "MOPoints"]
        for data in self.data:
            for pt in point_types:
                data[pt] = np.where(data[pt] > 600, data[pt] / 4, data[pt])
            data["AOMaxPoints"] = np.where(
                data["AOMaxPoints"] > 150, data["AOMaxPoints"] / 4, data["AOMaxPoints"]
            )

    def list_keys(self):
        """
        Prints all the keys from the dictionary stored in the second element of the data attribute.
        This method iterates over the keys of the dictionary located at index 1 of the `data` attribute
        and prints each key to the console.
        Returns:
            None
        """

        keys = []
        for key in self.data[1].keys():
            keys.append(key)
        return keys

    def return_annotators(self):
        """
        Extracts and returns a list of annotators from the dataset.
        Iterates through the dataset and collects the annotator information
        from each file, appending it to a list.
        Returns:
            list: A list of annotators as strings.
        """

        annotators = []
        for file in self.data:
            annotators.append(str(file["annotator"][0]))
        return annotators

    def view_beat_peaks(self, annotator, beat_num, trunc_percent=0, fig_size=(10, 6)):
        """
        Visualizes the SCG beat peaks and valleys for a given annotator and beat number.
        Parameters:
            annotator (str): The identifier for the annotator whose data is to be visualized.
            beat_num (int): The index of the beat to be visualized.
            trunc_percent (float, optional): The percentage of the beat to truncate from the end. Default is 0.
            fig_size (tuple, optional): The size of the figure to be plotted. Default is (10, 6).
        Returns:
            None
        Raises:
            Exception: If there is an error in plotting the beat, such as an index out of range.
        """

        # Find correct file for the given annotator
        for i, file in zip(range(len(self.data)), self.data):
            if str(file["annotator"][0]) == annotator:
                idx = i
        # Plot figure
        try:
            # Initalize fiducial points
            ao = self.data[idx]["AOMaxPoints"][:, beat_num] * 2
            ac = self.data[idx]["ACMaxPoints"][:, beat_num] * 2
            acv = self.data[idx]["ACvPoints"][:, beat_num] * 2
            mo = self.data[idx]["MOPoints"][:, beat_num] * 2

            # Create a new figure
            plt.figure(figsize=fig_size)
            scg_beat = self.data[idx]["scgBeats"][beat_num, :]

            # Find amps of fiducial points in beat
            ao_amp = scg_beat[int(ao)]
            ac_amp = scg_beat[int(ac)]
            acv_amp = scg_beat[int(acv)]
            mo_amp = scg_beat[int(mo)]

            # For truncating the figure
            end_idx = int(len(scg_beat) - ((trunc_percent / 100) * (len(scg_beat))))
            # Find peaks and valleys
            peaks, _ = find_peaks(scg_beat[:end_idx], distance=15)
            valleys, _ = find_peaks(-scg_beat[:end_idx], distance=15)
            # Plot figure
            plt.plot(scg_beat[:end_idx], label="SCG Signal")
            plt.plot(
                peaks, scg_beat[:end_idx][peaks], "x", color="tab:green", label="Peaks"
            )
            plt.plot(
                valleys,
                scg_beat[:end_idx][valleys],
                "x",
                color="tab:red",
                label="Valleys",
            )
            print(valleys)
            print(acv, mo)
            plt.scatter(ao, ao_amp, alpha=0.9, color="tab:blue", label="AO")
            plt.scatter(ac, ac_amp, alpha=0.9, color="tab:green", label="AC")
            plt.scatter(acv, acv_amp, alpha=0.9, color="tab:red", label="ACv")
            plt.scatter(mo, mo_amp, alpha=0.9, color="tab:purple", label="MO")
            plt.suptitle(
                f"Annotator:{annotator} - SCG Beat:{beat_num} - Peaks and Valleys Marked"
            )
            plt.grid()
            plt.legend()
            plt.show()
        except Exception as e:
            print(
                f"Error plotting beat. Index range 0 - {self.data['scgBeats'].shape[1]}"
            )

    def return_median_points(self, point_type, rating=2):
        """
        Returns the median points of a specified type from the dataset.
        Parameters:
            point_type (str): The type of points to return the median for.
                  Valid options are 'ao', 'ac', 'acv', 'mo'.
        Returns:
            numpy.ndarray: The median points of the specified type across all data entries.
        Raises:
            ValueError: If an invalid point_type is provided.
        """

        median_ratings = self.return_median_ratings()
        median_mask = median_ratings == rating

        if rating > 2:
            median_mask = median_ratings < rating
        point_type.lower()
        if point_type == "ao":
            sel = "AOMaxPoints"
        elif point_type == "ac":
            sel = "ACMaxPoints"
        elif point_type == "acv":
            sel = "ACvPoints"
        elif point_type == "mo":
            sel = "MOPoints"
        elif point_type == "shifted":
            sel = "ShiftedBeats"
        else:
            print("Invalid point_type... ('ao', 'ac', 'acv', 'mo')")
        points_list = np.squeeze(self.data[0][sel][:, median_mask])
        for i in range(len(self.data)):
            if i == 0:
                continue
            points_list = np.vstack(
                (points_list, np.squeeze(self.data[i][sel][:, median_mask]))
            )
        points_list = np.where(points_list == 0, np.nan, points_list)
        return np.nanmedian(points_list, axis=0)

    def return_median_points_all(self, point_type):
        """
        Returns the median points of a specified type from the dataset, regardless of rating.

        Parameters:
            point_type (str): The type of points to return the median for.
                              Valid options are 'ao', 'ac', 'acv', 'mo'.

        Returns:
            numpy.ndarray: The median points of the specified type across all data entries.

        Raises:
            ValueError: If an invalid point_type is provided.
        """
        point_type = point_type.lower()
        if point_type == "ao":
            sel = "AOMaxPoints"
        elif point_type == "ac":
            sel = "ACMaxPoints"
        elif point_type == "acv":
            sel = "ACvPoints"
        elif point_type == "mo":
            sel = "MOPoints"
        elif point_type == "shifted":
            sel = "ShiftedBeats"
        else:
            raise ValueError(
                "Invalid point_type. Valid options are: 'ao', 'ac', 'acv', 'mo'"
            )
        points_list = np.squeeze(self.data[0][sel])
        for i in range(1, len(self.data)):
            points_list = np.vstack((points_list, np.squeeze(self.data[i][sel])))

        points_list = np.where(points_list == 0, np.nan, points_list)
        return np.nanmedian(points_list, axis=0)

    def return_median_ratings(self):
        """
        Calculate and return the median ratings from the dataset.
        This method processes the 'ratings' field from each entry in the dataset,
        stacks them into a single array, and computes the median value along the
        specified axis.
        Returns:
            numpy.ndarray: An array containing the median ratings.
        """

        ratings = np.squeeze(self.data[0]["ratings"])
        for i in range(len(self.data)):
            if i == 0:
                continue
            ratings = np.vstack((ratings, np.squeeze(self.data[i]["ratings"])))
        return np.median(ratings, axis=0)

    def return_percentage_agreements(self, rating):
        median_ratings = self.return_median_ratings()
        ratings_mask = median_ratings >= rating
        masked_data = [{}, {}, {}]
        for i in range(3):
            masked_data[i]["scgBeats"] = self.data[i]["scgBeats"][ratings_mask, :]
            masked_data[i]["ecgBeats"] = self.data[i]["ecgBeats"][ratings_mask, :]
            masked_data[i]["AOMaxPoints"] = self.data[i]["AOMaxPoints"][:, ratings_mask]
            masked_data[i]["ACMaxPoints"] = self.data[i]["ACMaxPoints"][:, ratings_mask]
            masked_data[i]["ACvPoints"] = self.data[i]["ACvPoints"][:, ratings_mask]
            masked_data[i]["MOPoints"] = self.data[i]["MOPoints"][:, ratings_mask]

        if rating == 2 and float(self.file_number) in [2.1, 2.2, 2.3, 2.4]:
            return None, None, None, None
        else:
            ao_agree_percent = self.calculate_per_agreement(masked_data, "AOMaxPoints")
            ac_agree_percent = self.calculate_per_agreement_range(
                masked_data, "ACMaxPoints", 10
            )
            acv_agree_percent = self.calculate_per_agreement(masked_data, "ACvPoints")
            mo_agree_percent = self.calculate_per_agreement(masked_data, "MOPoints")
            return (
                float(ao_agree_percent),
                float(ac_agree_percent),
                float(acv_agree_percent),
                float(mo_agree_percent),
            )

    def calculate_per_agreement(self, masked_data, point_label):
        all_points_array = []
        for data in masked_data:
            points_array = np.squeeze(data[point_label])
            all_points_array.append(points_array)
        all_points_array = np.array(all_points_array, int).T
        annot_agree_0_1 = (all_points_array[:, 0] == all_points_array[:, 1]).astype(int)
        annot_agree_0_2 = (all_points_array[:, 0] == all_points_array[:, 2]).astype(int)
        annot_agree_1_2 = (all_points_array[:, 1] == all_points_array[:, 2]).astype(int)
        agreements = np.array((annot_agree_0_1, annot_agree_0_2, annot_agree_1_2)).T
        beat_agreements = np.sum(agreements, axis=1) * 1 / 3
        return np.sum(beat_agreements) / masked_data[0]["scgBeats"].shape[0]

    def calculate_per_agreement_range(self, masked_data, point_label, range):
        all_points_array = []
        for data in masked_data:
            points_array = np.squeeze(data[point_label])
            all_points_array.append(points_array)
        all_points_array = np.array(all_points_array, int).T
        annot_agree_0_1_lower = (
            all_points_array[:, 0] > (all_points_array[:, 1] - range)
        ).astype(int)
        annot_agree_0_1_upper = (
            all_points_array[:, 0] < (all_points_array[:, 1] + range)
        ).astype(int)
        annot_agree_0_1 = annot_agree_0_1_lower & annot_agree_0_1_upper

        annot_agree_0_2_lower = (
            all_points_array[:, 0] > (all_points_array[:, 2] - range)
        ).astype(int)
        annot_agree_0_2_upper = (
            all_points_array[:, 0] < (all_points_array[:, 2] + range)
        ).astype(int)
        annot_agree_0_2 = annot_agree_0_2_lower & annot_agree_0_2_upper

        annot_agree_1_2_lower = (
            all_points_array[:, 1] > (all_points_array[:, 2] - range)
        ).astype(int)
        annot_agree_1_2_upper = (
            all_points_array[:, 1] < (all_points_array[:, 2] + range)
        ).astype(int)
        annot_agree_1_2 = annot_agree_1_2_lower & annot_agree_1_2_upper

        agreements = np.array((annot_agree_0_1, annot_agree_0_2, annot_agree_1_2)).T
        beat_agreements = np.sum(agreements, axis=1) * 1 / 3
        return np.sum(beat_agreements) / masked_data[0]["scgBeats"].shape[0]

    def return_fleiss_kappa(self):
        """
        Calculate and return the Fleiss' kappa statistic for inter-rater reliability.
        This method processes the ratings data from multiple raters, aggregates it,
        and computes the Fleiss' kappa statistic to measure the agreement between
        the raters.
        Returns:
            float: The Fleiss' kappa statistic indicating the level of agreement
                   between the raters.
        """

        data = [
            np.squeeze(self.data[0]["ratings"]),
            np.squeeze(self.data[1]["ratings"]),
            np.squeeze(self.data[2]["ratings"]),
        ]
        data = np.array((data))
        data, cat = aggregate_raters(data.T)
        return fleiss_kappa(data)

    def return_krippendorff_alpha(self, rating):
        ratings = self.return_median_ratings()
        mask = ratings >= rating

        ao_arr = np.concatenate(
            [
                self.data[0]["AOMaxPoints"][:, mask],
                self.data[1]["AOMaxPoints"][:, mask],
                self.data[2]["AOMaxPoints"][:, mask],
            ]
        )
        if float(self.file_number) in [2.1, 2.2, 2.3, 2.4]:
            # print(ao_arr.shape)
            # print("ao_arr", ao_arr)
            if ao_arr.shape[1] == 0:
                return {"ao": None, "ac": None, "acv": None, "mo": None}
        ao_krip = krippendorff_alpha(
            ao_arr,
            metric=nominal_metric,
            force_vecmath=True,
            convert_items=int,
            missing_items="0",
        )

        if rating >= 2 and float(self.file_number) in [2.1, 2.2, 2.3, 2.4]:
            return {"ao": ao_krip, "ac": None, "acv": None, "mo": None}

        ac_arr = np.concatenate(
            [
                self.data[0]["ACMaxPoints"][:, mask],
                self.data[1]["ACMaxPoints"][:, mask],
                self.data[2]["ACMaxPoints"][:, mask],
            ]
        )
        acv_arr = np.concatenate(
            [
                self.data[0]["ACvPoints"][:, mask],
                self.data[1]["ACvPoints"][:, mask],
                self.data[2]["ACvPoints"][:, mask],
            ]
        )
        mo_arr = np.concatenate(
            [
                self.data[0]["MOPoints"][:, mask],
                self.data[1]["MOPoints"][:, mask],
                self.data[2]["MOPoints"][:, mask],
            ]
        )

        ac_krip = krippendorff_alpha(
            ac_arr,
            metric=nominal_metric,
            force_vecmath=True,
            convert_items=int,
            missing_items="0",
        )
        acv_krip = krippendorff_alpha(
            acv_arr,
            metric=nominal_metric,
            force_vecmath=True,
            convert_items=int,
            missing_items="0",
        )
        mo_krip = krippendorff_alpha(
            mo_arr,
            metric=nominal_metric,
            force_vecmath=True,
            convert_items=int,
            missing_items="0",
        )
        return {"ao": ao_krip, "ac": ac_krip, "acv": acv_krip, "mo": mo_krip}

    def view_median_points(self, save=False, vertical=False):
        """
        Visualizes the median points of SCG beats and their fiducial points.
        Parameters:
            save : bool, optional
                If True, saves the generated plots to a specified directory. Default is False.
            vertical : bool, optional
                If True, arranges the subplots vertically. Default is False.
        Returns:
            None
        """

        ao_points = self.return_median_points("ao")
        ac_points = self.return_median_points("ac")
        acv_points = self.return_median_points("acv")
        mo_points = self.return_median_points("mo")

        # TODO: display only beats that are good?
        # Find TRUE between all ratings
        ratings = []
        for data in self.data:
            ratings.append(np.squeeze(data["ratings"] == 2))

        ao_mask = ~np.isnan(ao_points)
        ac_mask = ~np.isnan(ac_points)
        acv_mask = ~np.isnan(acv_points)
        mo_mask = ~np.isnan(mo_points)

        # TODO: Apply ratings mask to scg_beats
        scg_beats = np.array(self.data[0]["scgBeats"])
        ao_points = (
            np.array([x for x, keep in zip(np.squeeze(ao_points), ao_mask) if keep]) * 2
        )
        ac_points = (
            np.array([x for x, keep in zip(np.squeeze(ac_points), ac_mask) if keep]) * 2
        )
        acv_points = (
            np.array([x for x, keep in zip(np.squeeze(acv_points), acv_mask) if keep])
            * 2
        )
        mo_points = (
            np.array([x for x, keep in zip(np.squeeze(mo_points), mo_mask) if keep]) * 2
        )

        scg_beats_new = scg_beats.T

        if not vertical:
            f, ax = plt.subplots(1, 3, figsize=(23, 6))
        else:
            f, ax = plt.subplots(3, 1, figsize=(8, 18))
        ax[0].imshow(scg_beats.T, cmap="gray", aspect="auto")
        ax[0].plot(ao_points.T, color="blue", alpha=0.8, label="AO")
        ax[0].plot(ac_points.T, color="g", alpha=0.8, label="AC")
        ax[0].plot(acv_points.T, color="r", alpha=0.8, label="ACv")
        ax[0].plot(mo_points.T, color="purple", alpha=0.8, label="MO")
        ax[0].set_ylabel("Time from R-peak [samples]")
        ax[0].set_xlabel("Beats")
        ax[0].set_title(
            "Heatmap Plot of SCG Beats and Fiducial Point Trends of Median Points"
        )
        ax[0].legend()

        ao_amp = [
            scg_beats_new[ao_loc_i, scg_beat_idx][0]
            for ao_loc_i, scg_beat_idx in zip(
                ao_points.astype(int), np.expand_dims(np.arange(len(scg_beats)), axis=1)
            )
        ]
        ac_amp = [
            scg_beats_new[ac_loc_i, scg_beat_idx][0]
            for ac_loc_i, scg_beat_idx in zip(
                ac_points.astype(int), np.expand_dims(np.arange(len(scg_beats)), axis=1)
            )
        ]
        acv_amp = [
            scg_beats_new[acv_loc_i, scg_beat_idx][0]
            for acv_loc_i, scg_beat_idx in zip(
                acv_points.astype(int),
                np.expand_dims(np.arange(len(scg_beats)), axis=1),
            )
        ]
        mo_amp = [
            scg_beats_new[mo_loc_i, scg_beat_idx][0]
            for mo_loc_i, scg_beat_idx in zip(
                mo_points.astype(int), np.expand_dims(np.arange(len(scg_beats)), axis=1)
            )
        ]
        ax[1].plot(scg_beats_new, color="gray", alpha=0.04)
        ax[1].plot(np.mean(scg_beats_new, axis=1), color="red", alpha=0.6, lw=3)
        ax[1].scatter(ao_points, ao_amp, alpha=0.6, color="tab:blue", label="AO")
        ax[1].scatter(ac_points, ac_amp, alpha=0.6, color="tab:green", label="AC")
        ax[1].scatter(acv_points, acv_amp, alpha=0.6, color="tab:red", label="ACv")
        ax[1].scatter(mo_points, mo_amp, alpha=0.6, color="tab:purple", label="MO")
        ax[1].set_xlabel("Time from R-peak [samples]")
        ax[1].set_ylabel("SCG amplitude")
        ax[1].set_title("Fiducial Points for Individual SCG Beats with Medians")
        ax[1].legend()

        ax[2].plot(np.mean(scg_beats_new, axis=1), color="red", alpha=0.6, lw=3)
        ax[2].scatter(
            np.mean(ao_points),
            np.mean(scg_beats_new, axis=1)[np.mean(ao_points).astype(int)],
            color="tab:blue",
            label="AO",
        )
        ax[2].scatter(
            np.mean(ac_points),
            np.mean(scg_beats_new, axis=1)[np.mean(ac_points).astype(int)],
            color="tab:green",
            label="AC",
        )
        ax[2].scatter(
            np.mean(acv_points),
            np.mean(scg_beats_new, axis=1)[np.mean(acv_points).astype(int)],
            color="tab:red",
            label="ACv",
        )
        ax[2].scatter(
            np.mean(mo_points),
            np.mean(scg_beats_new, axis=1)[np.mean(mo_points).astype(int)],
            color="tab:purple",
            label="MO",
        )
        ax[2].set_xlabel("Time from R-peak [samples]")
        ax[2].set_ylabel("SCG amplitude")
        ax[2].set_title("SCG Global Average with Median Locations of Fiducal Points")
        ax[2].legend()

        plt.suptitle(f"File {self.file_name}", y=0.95)

        if save:
            save_dir = config.FIGURES_DIR / "median-plots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/Hypovolemia_Median_Heatmaps_{self.file_name}.png")


def multirater_kfree(n_ij, n, k):
    """
    Computes Randolph's free marginal multirater kappa for assessing the
    reliability of agreement between annotators.

    Args:
        n_ij: An N x k array of ratings, where n_ij[i][j] annotators
              assigned case i to category j.
        n:    Number of raters.
        k:    Number of categories.
    Returns:
        Percentage of overall agreement and free-marginal kappa

    See also:
        http://justusrandolph.net/kappa/
    """
    N = len(n_ij)

    P_e = 1.0 / k
    P_O = (
        1.0
        / (N * n * (n - 1))
        * (sum(n_ij[i][j] ** 2 for i in xrange(N) for j in xrange(k)) - N * n)
    )

    kfree = (P_O - P_e) / (1 - P_e)

    return P_O, kfree
