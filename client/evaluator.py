import os
import argparse

from client import Client
from utils import answers_enricher
from metrics import get_all_metrics, get_aggregative_metrics


def parse_args():

    parser = argparse.ArgumentParser(description="Asks endpoint with the folder of files")

    parser.add_argument("--folder-path",
                        help="Path to the folder which contains files. Folder might have nested structure.",
                        required=True)

    parser.add_argument("--url",
                        help="Address of the endpoint to ask.",
                        required=True)

    parser.add_argument("--reports-path",
                        help="Path to the folder with reports.",
                        default="./reports")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # create an instance of the 'Client'
    # in this case we want to get answers for all the files in the specific folder
    # and store the results in corresponding folder
    tester = Client(in_dir=args.folder_path, out_path=args.reports_path, url=args.url)

    # query the endpoint for the results
    # answers will be saved to the specified path
    answers_folder = tester.query()

    # add ground truth to the predicted data
    _ = answers_enricher(os.path.join(answers_folder, "parsed_answers.csv"))

    # compute row-wise metrics
    _ = get_all_metrics(os.path.join(answers_folder, "parsed_answers.csv"))

    # compute aggregative metrics
    _ = get_aggregative_metrics(answers_folder)
