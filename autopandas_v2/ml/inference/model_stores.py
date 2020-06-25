import multiprocessing
import time
from multiprocessing.connection import Listener, Client
from typing import Dict, Any, List

import tqdm

from autopandas_v2.ml.inference.interfaces import RelGraphInterface


class ModelStore:
    def __init__(self, path_map: Dict[Any, str]):
        self.path_map = path_map
        self.model_store: Dict[str, RelGraphInterface] = {}
        self.cache = None
        self.setup()

    def setup(self):
        iterator = self.path_map.items()
        if len(self.path_map) > 3:
            iterator = tqdm.tqdm(iterator)

        for key, path in iterator:
            self.model_store[key] = RelGraphInterface.from_model_dir(path)

    def wait_till_ready(self):
        return True

    def predict_graphs(self, key, encodings: List[Dict], top_k: int = 10):
        return self.model_store[key].predict_graphs(encodings, top_k=top_k)

    def __contains__(self, item):
        return item in self.path_map

    def get_path_map(self):
        return self.path_map

    def close(self):
        for model in self.model_store.values():
            pass
            # model.close()

    def start_caching(self):
        self.cache = {}

    def stop_caching(self):
        self.cache = None


class ModelStoreServer(ModelStore):
    def __init__(self, path_map: Dict[Any, str], port: int = 6543):
        self.port = port
        self.process: multiprocessing.Process = None
        super().__init__(path_map)

    def setup(self):
        self.process = multiprocessing.Process(target=ModelStoreServer.serve, args=(self.path_map, self.port))
        self.process.start()

    def predict_graphs(self, key, encodings: List[Dict], **kwargs):
        conn = Client(('localhost', self.port))
        conn.send((key, encodings, kwargs))
        result = conn.recv()
        conn.close()
        return result

    def wait_till_ready(self):
        ready = False
        while not ready:
            try:
                conn = Client(('localhost', self.port))
                conn.close()
                ready = True
            except ConnectionRefusedError:
                time.sleep(2)

    @staticmethod
    def serve(path_map: Dict[Any, str], port: int):
        model_store = ModelStore(path_map)
        listener = Listener(('localhost', port))
        while True:
            conn = listener.accept()
            try:
                key, encodings, kwargs = conn.recv()
                if key == 'autopandas-exit':
                    #  Hacky AF
                    conn.close()
                    break

                conn.send(model_store.predict_graphs(key, encodings, **kwargs))
                conn.close()

            except:
                continue

            finally:
                conn.close()

        listener.close()

    def close(self):
        conn = Client(('localhost', self.port))
        conn.send(('autopandas-exit', [], {}))
        conn.close()
        self.process.join()
