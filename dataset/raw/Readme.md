# raw dataset and script
## description
This folder contains various original files and scripts used in the dataset creation process. This dataset consists of two parts: positive and negative.

## positive
Unzip positive_original.rar to obtain positive_original.jsonl.

positive_original.jsonl is obtained by filtering Python language data from The Pile's GitHub dataset.

positive.jsonl is derived from positive_original.jsonl: starting from the 1st line, take the first 10 lines out of every 100 lines. A total of 10,000 lines are processed, resulting in 1,000 lines.

abstract_script.py is the script used to generate positive.jsonl from positive_original.jsonl.

For the repository used for The Pile dataset, please refer to github_repositories.csv (from https://github.com/EleutherAI/github-downloader).

## negative
negative_raw.jsonl is obtained from GitHub: filter Python language repositories created after January 1, 2024, sort them in descending order of star count. After finding 10 Python functions in a repository, switch to next repository. A total of 100 repositories are included, resulting in 1,000 entries.

collect_script.py is the script used to collect negative_raw.jsonl.