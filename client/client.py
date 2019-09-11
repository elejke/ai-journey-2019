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
    Prior to sending to the endpoint the image is converted to the base64 encoded string
    in accordance with API specs.
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

        # create report folder
        # the name of the folder is the UNIX timestamp of the moment when class instance was created
        self._report_path = os.path.join(out_path, str(int(time.time() * 10 ** 6)))
        os.makedirs(self._report_path, exist_ok=False)

        # read input files
        if os.path.isdir(in_dir):
            # find all the files in the folder and its subfolders
            self.filelist = glob.glob(os.path.join(self.in_dir, "**/*.json"), recursive=True)
        else:
            raise ValueError("'in_dir' should be a directory!")

        self._is_ready()

    def _is_ready(self):
        num_retries = 0
        while num_retries < 600:
            try:
                response = requests.get(os.path.join(self.url, "ready"))
                if response.status_code == 200:
                    break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                time.sleep(1)
                num_retries += 1

    def query(self, save=True, n_jobs=1) -> pd.DataFrame:
        """ Queries the endpoint with all the files specified in 'in_dir' folder.
        This method goes over the list of files, sends every of them to the specified endpoint and
        put all the results into one DataFrame.
        Each column of the dataframe corresponds to the first-level key in the response json.
        If the json dict is nested then the values of the cells would be dicts themselves.
        In case of failure in one of the requests the corresponding line would contain all NaN.
        Args:
            save (bool): Whether to save the results of the query or not.
            n_jobs (int): number of processes to use for querying.
        Return:
            Dataframe with the results of the requests for all the images.
        """

        def get_one_answer(file):
            return json.dumps(ask_endpoint(file, os.path.join(self.url, "take_exam")))

        # send each file to the endpoint
        answers = Parallel(n_jobs=n_jobs)(delayed(get_one_answer)(file) for file in tqdm(self.filelist))

        # put all answers to the dataframe
        answers = pd.DataFrame(answers, columns=["predictions"])
        answers["path"] = self.filelist

        if save:
            answers.to_csv(os.path.join(self._report_path, "answers.csv"), index=False)

        return answers
