"""Download the CIFAR-10 python batches (for the six image statistics) into $IB_DATA_SNN/cifar."""
import argparse
import os
import tarfile
import time
import urllib.request

from snn_common import CIFAR_SUBDIR

from ib.paths import data_dir

URLS = ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"]


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cif = data_dir("snn") / CIFAR_SUBDIR
    cif.mkdir(parents=True, exist_ok=True)
    if (cif / "cifar-10-batches-py" / "test_batch").exists():
        print(f"CIFAR-10 already present in {cif}")
        return
    tgz = cif / "c10.tar.gz"
    for attempt in range(6):
        url = URLS[attempt % len(URLS)]
        try:
            print(f"attempt {attempt}: {url}", flush=True)
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tgz, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"downloaded {os.path.getsize(tgz) / 1e6:.0f} MB in {time.time() - t0:.0f}s")
            with tarfile.open(tgz) as t:
                t.extractall(cif)
            assert (cif / "cifar-10-batches-py" / "test_batch").exists()
            print(f"[OK] CIFAR-10 ready in {cif}")
            return
        except Exception as e:
            print("failed:", e, flush=True)
            if tgz.exists():
                tgz.unlink()
            time.sleep(5)
    raise SystemExit("[FAILED] could not download CIFAR-10")


if __name__ == "__main__":
    main()
