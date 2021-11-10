import os
import pickle
from datetime import datetime
from json import dumps
from typing import Dict, List, NamedTuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from kafka import KafkaProducer
from pydantic import BaseModel

from gamma import BayesianGaussianMixture, GaussianMixture
from gamma.utils import association, convert_picks_csv, from_seconds, to_seconds

try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka-headless:9092'
    producer = KafkaProducer(
        bootstrap_servers=[BROKER_URL],
        key_serializer=lambda x: dumps(x).encode('utf-8'),
        value_serializer=lambda x: dumps(x).encode('utf-8'),
    )
    print('k8s kafka connection success!')
except BaseException:
    print('k8s Kafka connection error')

    try:
        print('Connecting to local kafka')
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            key_serializer=lambda x: dumps(x).encode('utf-8'),
            value_serializer=lambda x: dumps(x).encode('utf-8'),
        )
        print('local kafka connection success!')
    except BaseException:
        print('local Kafka connection error')

app = FastAPI()

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PKL = os.path.join(PROJECT_ROOT, "tests/config_hawaii.pkl")
STATION_CSV = os.path.join(PROJECT_ROOT, "tests/stations_hawaii.csv")

with open(CONFIG_PKL, "rb") as fp:
    config = pickle.load(fp)
## read stations
stations = pd.read_csv(STATION_CSV, delimiter="\t")
stations = stations.rename(columns={"station": "id"})
stations["x(km)"] = stations["longitude"].apply(lambda x: (x - config["center"][0]) * config["degree2km"])
stations["y(km)"] = stations["latitude"].apply(lambda x: (x - config["center"][1]) * config["degree2km"])
stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)
## setting GMMA configs
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["use_dbscan"] = True
config["use_amplitude"] = True
config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"]
config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
config["z(km)"] = (0, 40)
# DBSCAN
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (None, None),
)  # t
config["dbscan_eps"] = min(
    np.sqrt(
        (stations["x(km)"].max() - stations["x(km)"].min()) ** 2
        + (stations["y(km)"].max() - stations["y(km)"].min()) ** 2
    )
    / (6.0 / 1.75),
    10,
)  # s
config["dbscan_min_samples"] = min(len(stations), 3)
# Filtering
config["min_picks_per_eq"] = min(len(stations) // 2, 10)
config["oversample_factor"] = min(len(stations) // 2, 10)
for k, v in config.items():
    print(f"{k}: {v}")


class Data(BaseModel):
    picks: List[Dict[str, Union[float, str]]]
    stations: List[Dict[str, Union[float, str]]]
    config: Dict[str, Union[List[float], float, str]]

class Pick(BaseModel):
    picks: List[Dict[str, Union[float, str]]]

@app.get('/predict_stream')
def predict(data: Pick):

    picks = data.picks
    if len(picks) == 0:
        return []

    # picks = pd.read_json(picks)
    picks = pd.DataFrame(picks)
    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

    event_idx0 = 0
    if (len(picks) > 0) and (len(picks) < 5000):
        data, locs, phase_type, phase_weight, phase_index = convert_picks_csv(picks, stations, config)
        catalogs, _ = association(data, locs, phase_type, phase_weight, len(stations), phase_index, event_idx0, config)
        event_idx0 += len(catalogs)
    else:
        catalogs = []
        picks["time_idx"] = picks["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H"))  ## process by hours
        for hour in sorted(list(set(picks["time_idx"]))):
            picks_ = picks[picks["time_idx"] == hour]
            if len(picks_) == 0:
                continue
            data, locs, phase_type, phase_weight, phase_index = convert_picks_csv(picks_, stations, config)
            catalog, _ = association(data, locs, phase_type, phase_weight, len(stations), phase_index, event_idx0, config)
            event_idx0 += len(catalog)
            catalogs.extend(catalog)

    ### create catalog
    catalogs = pd.DataFrame(catalogs, columns=["time(s)"] + config["dims"] + ["magnitude", "covariance"])
    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: x / config["degree2km"] + config["center"][0])
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: x / config["degree2km"] + config["center"][1])
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x * 1e3)
    # catalogs["event_idx"] = range(event_idx0)
    if config["use_amplitude"]:
        catalogs["covariance"] = catalogs["covariance"].apply(lambda x: f"{x[0][0]:.3f},{x[1][1]:.3f},{x[0][1]:.3f}")
    else:
        catalogs["covariance"] = catalogs["covariance"].apply(lambda x: f"{x[0][0]:.3f}")

    catalogs = catalogs[['time', 'magnitude', 'longitude', 'latitude', 'depth(m)', 'covariance']]
    catalogs = catalogs.to_dict(orient='records')
    print("GMMA:", catalogs)
    for event in catalogs:
        producer.send('gmma_events', key=event["time"], value=event)

    return catalogs


def default_config(config):
    if "degree2km" not in config:
        config["degree2km"] = 111.195
    if "use_amplitude" not in config:
        config["use_amplitude"] = True
    if "use_dbscan" not in config:
        config["use_dbscan"] = True
    if "dbscan_eps" not in config:
        config["dbscan_eps"] = 6
    if "dbscan_min_samples" not in config:
        config["dbscan_min_samples"] = 3
    if "oversample_factor" not in config:
        config["oversample_factor"] = 10
    if "min_picks_per_eq" not in config:
        config["min_picks_per_eq"] = 10
    if "dims" not in config:
        config["dims"] = ["x(km)", "y(km)", "z(km)"]
    return config


@app.post('/predict')
def predict(data: Data):

    config = data.config
    stations = pd.DataFrame(data.stations)
    picks = pd.DataFrame(data.picks)
    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    config = default_config(config)
    assert("latitude" in stations)
    assert("longitude" in stations)
    assert("elevation(m)" in stations)
    
    if "xlim_degree" not in config:
        config["xlim_degree"] = (stations["longitude"].min(), stations["longitude"].max())
    if "ylim_degree" not in config:
        config["ylim_degree"] = (stations["latitude"].min(), stations["latitude"].max())
    if "center" not in config:
        config["center"] = [np.mean(config["xlim_degree"]), np.mean(config["ylim_degree"])]
    if "x(km)" not in config:
        config["x(km)"] = (np.array(config["xlim_degree"]) - config["center"][0])*config["degree2km"]
    if "y(km)" not in config:
        config["y(km)"] = (np.array(config["ylim_degree"]) - config["center"][1])*config["degree2km"]
    if "z(km)" not in config:
        config["z(km)"] = (0, 41)
    if "bfgs_bounds" not in config:
        config["bfgs_bounds"] = [list(config[x]) for x in config["dims"]] + [[None, None]]

    stations["x(km)"] = stations["longitude"].apply(lambda x: (x - config["center"][0]) * config["degree2km"])
    stations["y(km)"] = stations["latitude"].apply(lambda x: (x - config["center"][1]) * config["degree2km"])
    stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)

    if len(picks) == 0:
        return []

    event_idx0 = 0 ## current earthquake index
    assignments = []
    if (len(picks) > 0) and (len(picks) < 5000):
        data, locs, phase_type, phase_weight, phase_index = convert_picks_csv(picks, stations, config)
        catalogs, assignments = association(data, locs, phase_type, phase_weight, len(stations), phase_index, event_idx0, config)
        event_idx0 += len(catalogs)
    else:
        catalogs = []
        picks["time_idx"] = picks["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H")) ## process by hours
        for hour in sorted(list(set(picks["time_idx"]))):
            picks_ = picks[picks["time_idx"] == hour]
            if len(picks_) == 0:
                continue
            data, locs, phase_type, phase_weight, phase_index = convert_picks_csv(picks_, stations, config)
            catalog, assign = association(data, locs, phase_type, phase_weight, len(stations), phase_index, event_idx0, config)
            event_idx0 += len(catalog)
            catalogs.extend(catalog)
            assignments.extend(assign)

    ## create catalog
    catalogs = pd.DataFrame(catalogs, columns=["time(s)"]+config["dims"]+["magnitude", "covariance"])
    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: x/config["degree2km"] + config["center"][0])
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: x/config["degree2km"] + config["center"][1])
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x*1e3)
    catalogs["event_idx"] = range(event_idx0)
    if config["use_amplitude"]:
        catalogs["covariance"] = catalogs["covariance"].apply(lambda x: f"{x[0][0]:.3f},{x[1][1]:.3f},{x[0][1]:.3f}")
    else:
        catalogs["covariance"] = catalogs["covariance"].apply(lambda x: f"{x[0][0]:.3f}")
    catalogs.drop(columns=["x(km)", "y(km)", "z(km)", "time(s)"], inplace=True)

    ## add assignment to picks
    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gmma"])
    picks_gamma = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({'event_idx': int})
    picks_gamma["timestamp"] = picks_gamma["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
    if "time_idx" in picks_gamma:
        picks_gamma.drop(columns=["time_idx"], inplace=True)

    return {"catalog": catalogs.to_json(orient="records"), 
            "picks": picks_gamma.to_json(orient="records")}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}