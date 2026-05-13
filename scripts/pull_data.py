#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

PROTOC_URL = "https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip"
REMOTE_DEFAULT = "/storage/emulated/0/Android/data/se.lth.math.videoimucapture/files"

def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)

def _check_adb() -> None:
    try:
        out = subprocess.check_output(["adb", "devices"], text=True)
    except FileNotFoundError as exc:
        raise SystemExit("adb not found. Install Android platform-tools first.") from exc
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    device_lines = [x for x in lines[1:] if "\tdevice" in x]
    if not device_lines:
        raise SystemExit("No adb device connected in 'device' state.")

def _adb_shell(cmd: str) -> str:
    return subprocess.check_output(["adb", "shell", cmd], text=True).strip()

def _list_remote_children(remote_dir: str) -> list[str]:
    out = _adb_shell(f'ls -1 "{remote_dir}"')
    names = [x.strip() for x in out.splitlines() if x.strip()]
    return [x for x in names if x not in (".", "..")]

def _choose_capture_path(remote_dir: str) -> tuple[str, str]:
    children = _list_remote_children(remote_dir)
    if not children:
        raise SystemExit(f"No entries found under remote path: {remote_dir}")
    has_pb3_here = "video_meta.pb3" in children
    has_mp4_here = "video_recording.mp4" in children
    if has_pb3_here and has_mp4_here:
        return remote_dir, "capture_from_files_root"
    ts_pat = re.compile(r"^\d{4}[_-]\d{2}[_-]\d{2}[_-]\d{2}[_-]\d{2}[_-]\d{2}$")
    candidate_dirs = sorted([x for x in children if ts_pat.match(x)])
    if not candidate_dirs:
        candidate_dirs = sorted(children)
    chosen = candidate_dirs[-1]
    return f"{remote_dir.rstrip('/')}/{chosen}", chosen

def _adb_pull(remote_path: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    _run(["adb", "pull", remote_path, str(local_dir)])
    pulled_name = Path(remote_path).name
    local_capture = local_dir / pulled_name
    return local_capture

def _resolve_protoc(tools_dir: Path) -> Path:
    protoc_in_path = shutil.which("protoc")
    if protoc_in_path: return Path(protoc_in_path)
    tools_dir.mkdir(parents=True, exist_ok=True)
    protoc_root = tools_dir / "protoc-3.13.0"
    protoc_bin = protoc_root / "bin" / "protoc"
    if protoc_bin.exists(): return protoc_bin
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "protoc.zip"
        urllib.request.urlretrieve(PROTOC_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(protoc_root)
    return protoc_bin

def _install_python_deps() -> None:
    print("Installing python dependencies: protobuf, pyquaternion...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf", "pyquaternion"])

def _build_proto_module(videoimu_repo: Path, parser_out_dir: Path, protoc_bin: Path) -> None:
    proto_file = videoimu_repo / "protobuf" / "recording.proto"
    parser_out_dir.mkdir(parents=True, exist_ok=True)
    _run([str(protoc_bin), f"-I{videoimu_repo}", f"--python_out={parser_out_dir}", str(proto_file)])

def _load_recording_pb2(parser_out_dir: Path):
    sys.path.insert(0, str(parser_out_dir))
    sys.path.insert(0, str(parser_out_dir / "protobuf"))
    try:
        try:
            import recording_pb2 as pb2
        except ModuleNotFoundError:
            from protobuf import recording_pb2 as pb2
    finally:
        if str(parser_out_dir / "protobuf") in sys.path: sys.path.remove(str(parser_out_dir / "protobuf"))
        if str(parser_out_dir) in sys.path: sys.path.remove(str(parser_out_dir))
    return pb2

def _parse_pb3_capture(pb3_path: Path, pb2_module):
    data = pb2_module.VideoCaptureData()
    data.ParseFromString(pb3_path.read_bytes())
    t0_ns = int(data.time.seconds) * 1_000_000_000 + int(data.time.nanos)
    imu_rows = []
    for s in data.imu:
        if len(s.gyro) < 3 or len(s.accel) < 3: continue
        imu_rows.append((int(s.time_ns), (float(s.gyro[0]), float(s.gyro[1]), float(s.gyro[2])), (float(s.accel[0]), float(s.accel[1]), float(s.accel[2]))))
    frame_meta = [(int(v.frame_number), int(v.time_ns)) for v in data.video_meta]
    t_ref = min([imu_rows[0][0]] + ([min(t for _, t in frame_meta)] if frame_meta else []))
    imu_epoch = [(t0_ns + (t - t_ref), g, a) for t, g, a in imu_rows]
    frame_epoch = [(fn, t0_ns + (t - t_ref)) for fn, t in frame_meta]
    return t0_ns, imu_epoch, frame_epoch

def _write_imu_csv(imu_epoch, out_dir: Path) -> None:
    with (out_dir / "Gyroscope.csv").open("w", newline="") as fg, (out_dir / "Accelerometer.csv").open("w", newline="") as fa:
        wg, wa = csv.writer(fg), csv.writer(fa)
        wg.writerow(["time", "seconds_elapsed", "z", "y", "x"])
        wa.writerow(["time", "seconds_elapsed", "z", "y", "x"])
        t0 = imu_epoch[0][0]
        for t_ns, gyro, accel in imu_epoch:
            sec = (t_ns - t0) * 1e-9
            wg.writerow([t_ns, sec, gyro[2], gyro[1], gyro[0]])
            wa.writerow([t_ns, sec, accel[2], accel[1], accel[0]])

def _extract_camera_frames(video_path: Path, out_dir: Path, t0_ns: int, frame_epoch) -> int:
    camera_dir = out_dir / "Camera"
    camera_dir.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video_path), "-vsync", "0", str(camera_dir / "frame_%06d.jpg")])
    extracted = sorted(camera_dir.glob("frame_*.jpg"))
    frame_meta_sorted = sorted(frame_epoch, key=lambda x: x[0])
    for i, frame_path in enumerate(extracted):
        t_ns = frame_meta_sorted[i][1] if i < len(frame_meta_sorted) else t0_ns + i * 33_333_333
        frame_path.rename(camera_dir / f"{t_ns // 1000000}.jpg")
    return len(extracted)

def _build_proto_module(proto_file: Path, parser_out_dir: Path, protoc_bin: Path) -> None:
    parser_out_dir.mkdir(parents=True, exist_ok=True)
    # Note: -I must point to the directory containing the proto file
    _run([str(protoc_bin), f"-I{proto_file.parent}", f"--python_out={parser_out_dir}", str(proto_file)])

def main() -> None:
    orb_lite_root = Path(__file__).resolve().parents[1]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(orb_lite_root / "data"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    _check_adb()
    remote_capture, _ = _choose_capture_path(REMOTE_DEFAULT)
    local_capture = _adb_pull(remote_capture, out_dir)
    
    # --- NEW: Self-Contained Proto Logic ---
    scripts_dir = orb_lite_root / "scripts"
    proto_file = scripts_dir / "recording.proto"
    
    # Download the proto file on the fly if it's missing
    if not proto_file.exists():
        print(f"Downloading recording.proto to {proto_file}...")
        url = "https://raw.githubusercontent.com/DavidGillsjo/VideoIMUCapture-Android/master/protobuf/recording.proto"
        urllib.request.urlretrieve(url, proto_file)
        
    parser_out_dir = scripts_dir / "proto_python"
    protoc_bin = _resolve_protoc(orb_lite_root / ".tools")
    
    _build_proto_module(proto_file, parser_out_dir, protoc_bin)
    # ---------------------------------------
    
    _install_python_deps()
    
    pb2 = _load_recording_pb2(parser_out_dir)
    t0_ns, imu_epoch, frame_epoch = _parse_pb3_capture(local_capture / "video_meta.pb3", pb2)
    _write_imu_csv(imu_epoch, local_capture)
    _extract_camera_frames(local_capture / "video_recording.mp4", local_capture, t0_ns, frame_epoch)
    print(f"Done. Data in {local_capture}")

if __name__ == "__main__":
    main()
