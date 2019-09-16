import os
import time
import json
import glob
import requests

import pandas as pd

from tqdm import tqdm

from joblib import Parallel, delayed


def ask_endpoint(file, endpoint) -> dict:
    """ Take the file and send it to the endpoint.
    Args:
        file (str or dict): Path to the input file or already loaded json.
        endpoint (str): URL of the endpoint.
    Return:
        Response from the endpoint with the results.
        If the error has occurred during processing then the empty dict is returned.
    """

    # define headers with the content type
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = None
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    elif isinstance(file, dict):
        data = file
    else:
        raise TypeError("'file' should be either file path or loaded json!")

    # send request to the endpoint
    response = requests.post(endpoint,
                             json=data,
                             headers=headers)

    # check response status and parse the body
    if response.status_code == 200:
        response = response.json()
    else:
        response = {}

    return response


class Client(object):
    """ Client for working with the AI Journey API.
    It allows to get the answers for all the files in the directory.
    """

    def __init__(self, in_dir, out_path, url):
        """ Constructor.
        Args:
            in_dir (str): Path to the directory with files to process. It is scanned recursively.
            out_path (str): Path to the folder where to store the results.
            url(str): URL of the endpoint.
        """

        # set class attributes
        self.in_dir = in_dir
        self.out_path = out_path
        self.url = url
        self.max_wait_time_ready = 120

        # the name of the folder is the UNIX timestamp of the moment when class instance was created
        self._report_path = os.path.join(out_path, str(int(time.time() * 10 ** 6)) +
                                         "_" +
                                         self.in_dir.strip("/").split("/")[-1])

        # read input files
        if os.path.isdir(in_dir):
            # find all the files in the folder and its subfolders
            self.filelist = glob.glob(os.path.join(self.in_dir, "**/*.json"), recursive=True)
        else:
            raise ValueError("'in_dir' should be a directory!")

        readiness_start_time = time.time()
        self._is_ready()
        readiness_end_time = time.time()
        self._readiness_time = readiness_end_time - readiness_start_time

    def _is_ready(self):
        """ Check that the endpoint is ready to receive income messages.
        """
        current_wait_time = 0
        start_time = time.time()
        while current_wait_time < self.max_wait_time_ready:
            try:
                response = requests.get(os.path.join(self.url, "ready"))
                if response.status_code == 200:
                    break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                time.sleep(1)
                current_wait_time = time.time() - start_time
        if current_wait_time >= self.max_wait_time_ready:
            raise TimeoutError("Interrupting execution\n'/ready' endpoint is not ready " +
                               "for maximum allowed {:d} seconds!".format(self.max_wait_time_ready))

    def query(self, n_jobs=1) -> str:
        """ Queries the endpoint with all the files specified in 'in_dir' folder.
        This method goes over the list of files, sends every of them to the specified endpoint and
        put all the results into one DataFrame.
        Each column of the dataframe corresponds to the first-level key in the response json.
        If the json dict is nested then the values of the cells would be dicts themselves.
        In case of failure in one of the requests the corresponding line would contain all NaN.
        Args:
            n_jobs (int): number of processes to use for querying.
        """

        def get_one_answer(file):
            return json.dumps(ask_endpoint(file, os.path.join(self.url, "take_exam")))

        # send each file to the endpoint
        query_start_time = time.time()
        answers = Parallel(n_jobs=n_jobs)(delayed(get_one_answer)(file) for file in tqdm(self.filelist))
        query_end_time = time.time()
        query_time = query_end_time - query_start_time

        # put all answers to the dataframe
        answers = pd.DataFrame(answers, columns=["prediction"])
        answers["prediction"] = answers["prediction"].apply(lambda x: json.loads(x))
        answers["path"] = self.filelist

        # create report folder
        os.makedirs(self._report_path, exist_ok=False)
        # save raw answers
        answers.to_csv(os.path.join(self._report_path, "raw_answers.csv"), index=False)
        # parse answers
        parsed_answers = pd.DataFrame(columns=["path",
                                               "id",
                                               "prediction"])
        for _, row in answers.iterrows():
            for k, v in row["prediction"]["answers"].items():
                parsed_answers.loc[len(parsed_answers)] = [row["path"], int(k), v]
        # save parsed answers
        parsed_answers = parsed_answers.sort_values(by=["path", "id"]).reset_index(drop=True)
        parsed_answers.to_csv(os.path.join(self._report_path, "parsed_answers.csv"), index=False)
        # save statistics
        stats = {
            "readiness_time": self._readiness_time,
            "query_total_files": len(self.filelist),
            "query_total_time": query_time,
            "query_n_jobs": n_jobs,
            "query_mean_latency": query_time / len(self.filelist) * n_jobs,
            "query_rps": len(self.filelist) / query_time
        }
        with open(os.path.join(self._report_path, "stats.json"), "w") as f:
            json.dump(stats, f)

        return self._report_path
