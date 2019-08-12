import json
import os
import pandas as pd
import subprocess
import sys
import time

from io import StringIO
from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import ArgNamespace


class GDriveRunner:
    def __init__(self, home_dir, cmd_args: ArgNamespace):
        self.home_dir = home_dir
        self.cmd_args = cmd_args
        self.max_gdrive_retries = cmd_args.max_gdrive_retries

        self.path_cache = {
            '/Data': '1hlg3OcR3uPiqJQRVPuLqJeeeB6R4ESyY',
            '/Data/Raw': '1vYcDRjSSzi6oIPpvOvW6PZOMBKzt7UdG',
            '/Data/Functions': '1uDf8Udvtz_F4aSpXZyouCDmIwZr2xxnW',
            '/Data/Generators': '1gonGvuvyPSu5LlLWWMutmb2-fZfdJHXo',
            '/Data/Raw/Pandas': '1rYbvhHqzH9FEAKRwHlJx2Sw_rrVPon8V',
            '/Data/Functions/Pandas': '1T1GBdH4AOL4Gl64A5ZXFxQxsX26sq3z6',
            '/Data/Generators/Pandas': '1JtF8lBIhZSzalgST1hEYlldjvs9WrCUc',
        }

        if not os.path.exists(home_dir + '/.gdrive/path_cache.json'):
            self.save_path_cache()
        else:
            with open(home_dir + '/.gdrive/path_cache.json', 'r') as f:
                self.path_cache.update(json.load(f))

    def save_path_cache(self):
        with open(self.home_dir + '/.gdrive/path_cache.json', 'w') as f:
            json.dump(self.path_cache, f)

    def get_id(self, path):
        if path in self.path_cache:
            return self.path_cache[path]

        basename = os.path.basename(path)
        cmd = "{home_dir}/gdrive list --name-width 0 " \
              "--absolute --query \"trashed = false and name contains '{path}'\"".format(home_dir=self.home_dir,
                                                                                         path=basename)
        listing = self.get_output(cmd)
        listing = pd.read_csv(StringIO(listing), delimiter=r'\s\s+')
        for g_id, g_name in zip(listing.Id, listing.Name):
            if g_name.endswith(path):
                self.path_cache[path] = g_id
                self.save_path_cache()
                return g_id

        raise Exception("Could not find path {path}".format(path=path))

    def get_output(self, cmd: str):
        attempts = 0
        sleep_time = 5
        max_sleep_time = 20
        while True:
            attempts += 1
            try:
                out = subprocess.check_output(cmd, shell=True)
                return out.decode("utf-8")

            except subprocess.CalledProcessError as e:
                e.output = str(e.output)
                if 'rateLimitExceeded' in e.output and attempts <= self.max_gdrive_retries:
                    logger.info("Rate Limit Exceeded. Waiting {sleep} seconds...".format(sleep=sleep_time))
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time + 5, max_sleep_time)
                    continue

                logger.err("Command {cmd} failed with exit code {code} "
                           "and output {output}".format(cmd=cmd, code=e.returncode, output=e.output))
                sys.exit(1)

    def run(self, cmd: str):
        attempts = 0
        sleep_time = 5
        max_sleep_time = 20
        code = os.system(cmd)
        while code != 0:
            attempts += 1
            if attempts <= self.max_gdrive_retries:
                logger.info("Retrying after {sleep} seconds...".format(sleep=sleep_time))
                time.sleep(sleep_time)
                sleep_time = min(sleep_time + 5, max_sleep_time)
                code = os.system(cmd)

                continue

            logger.err("Command {cmd} failed with exit code {code}".format(cmd=cmd, code=code))
            sys.exit(1)
