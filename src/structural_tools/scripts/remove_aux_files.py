import subprocess
import time


def run_rm() -> None:
    cmd = ["rm", "-f", "*.{aux,fdb_latexmk,fls,log,out,toc,synctex.gz}"]

    start = time.perf_counter()

    subprocess.run(
        cmd,
        check=True,
    )

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.1f}s")


def main():
    run_rm()


if __name__ == "__main__":
    main()
