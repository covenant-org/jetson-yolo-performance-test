import os
import mlflow
import argparse
import time
from multiprocessing import Process, Event, Value

TRACKING_URI = "https://mlflow.nuclea.solutions"

def avg_gpu_usg(event, result, max_gpu):
    from jtop import jtop
    sample_count=0
    total=0
    with jtop() as jetson:
        while jetson.ok() and not event.is_set():
            if jetson.stats['GPU'] > 0:
                if jetson.stats['GPU'] > max_gpu.value:
                    max_gpu.value = jetson.stats['GPU']
                sample_count+=1
                total += jetson.stats['GPU']
    result.value=total/sample_count


class TestFPSResult:
    def __init__(self):
        self.avg_fps = 0
        self.frame_count = 0
        self.time_elapsed = 0
        self.time_start = time.time()
        self.time_end = self.time_start
        self.experiment_name = ""
        self.run_name = ""
        self.avg_gpu_usage = 0
        self.max_gpu_usage = 0
        self.url = ""


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", action="extend",
                        nargs="+", type=str, help="Experiments (names) to download models from")
    parser.add_argument("--video", type=str,
                        default="test.mp4", help="Video to process")
    parser.add_argument("--output", type=str,
                        default="fps.csv", help="Output file in csv format")
    return parser.parse_args()


def get_video_capture(video_path):
    if not video_path:
        return None
    if not os.path.exists(video_path):
        return None
    import cv2
    return cv2.VideoCapture(video_path)


def test_model(model_path, video_path):
    import torch
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        print(f"{color.RED}!!Model {model_path} not found{color.END}")
        return None
    video = get_video_capture(video_path)
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    if not video.isOpened():
        print(f"{color.RED}!!Video {video_path} could not be opened{color.END}")
        return None
    results = TestFPSResult()
    results.frame_count = 0
    total_fps = 0
    finish_gpu_monitor = Event()
    gpu_usage = Value('d', 0.0)
    gpu_max_usage = Value('d', 0.0)
    gpu_monitor = Process(target=avg_gpu_usg, args=(finish_gpu_monitor,gpu_usage, gpu_max_usage))
    gpu_monitor.start()
    while True:
        start_time = time.time()
        ret, frame = video.read()
        if not ret:
            break
        results.frame_count += 1
        predictions = model.predict(source=frame, device=device, verbose=False)
        # To take into account the time it takes to process the frame
        annotated_frame = predictions[0].plot()

        end_time = time.time()
        frame_fps = 1 / (end_time - start_time)
        total_fps += frame_fps
        print(f"{color.GRAY}FPS: {frame_fps:.2f}{color.END}")
    finish_gpu_monitor.set()
    results.time_end = time.time()
    gpu_monitor.join()
    results.avg_gpu_usage = gpu_usage.value
    results.max_gpu_usage = gpu_max_usage.value
    results.avg_fps = total_fps / results.frame_count
    results.time_elapsed = results.time_end - results.time_start
    video.release()
    print(f"{color.GREEN}SUCCESS: FPS: {results.avg_fps} GPU: {results.avg_gpu_usage}{color.END}")
    return results


def save_to_csv(result: TestFPSResult, output_file):
    output_file.write(
        f"{result.experiment_name},{result.run_name},{result.avg_fps},{result.frame_count},{result.time_elapsed},{result.avg_gpu_usage},{result.max_gpu_usage},{result.url}\n")

def download_model(client, path_to_model, experiment, run):
    print(
        f"{color.YELLOW}Downloading model from run {run.info.run_name}{color.END}")
    client.download_artifacts(run_id=run.info.run_id, path=f"weights/best.pt",
                              dst_path=f"models/{experiment.name}")
    os.rename(
        f"models/{experiment.name}/weights/best.pt", path_to_model)
    os.rmdir(f"models/{experiment.name}/weights")


def main():
    args = get_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.client.MlflowClient(TRACKING_URI)
    output_file = open(args.output, "w")
    output_file.write("Experiment,Run,Avg FPS,Frames,Time Elapsed,Avg GPU Usage,Max GPU Usage,MLFLOW URL\n")
    for experiment in args.experiments:
        print(f"{color.BOLD}Searching runs from experiment {experiment}{color.END}")
        experiment = client.get_experiment_by_name(experiment)
        os.makedirs(exist_ok=True, name=f"models/{experiment.name}")
        if not experiment:
            print(f"{color.RED}!!Experiment {experiment} not found {color.END}")
            continue
        for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
            path_to_model = f"models/{experiment.name}/{run.info.run_name}.pt"
            if os.path.exists(path_to_model):
                print(
                    f"{color.GRAY}Model {run.info.run_name} already exists{color.END}")
            else:
                download_model(client, path_to_model, experiment, run)

            print(f"{color.CYAN}Running model {run.info.run_name}{color.END}")
            result = test_model(path_to_model, args.video)
            if result is None:
                print(f"{color.RED}!!Model {run.info.run_name} failed{color.END}")
                continue
            result.experiment_name = experiment.name
            result.run_name = run.info.run_name
            result.url = f"{TRACKING_URI}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
            save_to_csv(result, output_file)
        print("")
    output_file.close()


if __name__ == "__main__":
    main()
