import datetime
import glob
import logging
import subprocess
import tarfile


def create_submit_pkg() -> None:
    # Source files
    src_files = glob.glob("src/*.py")

    # Notebooks
    notebooks = glob.glob("*.ipynb")

    # Genereate HTML files from the notebooks
    for nb in notebooks:
        cmd_line = f"jupyter nbconvert --to html {nb}"

        logging.info(f"executing: {cmd_line}")
        subprocess.check_call(cmd_line, shell=True)

    html_files = glob.glob("*.htm*")

    now = datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h") + "m"
    outfile = f"submission_{now}.tar.gz"
    logging.info(f"Adding files to {outfile}")
    with tarfile.open(outfile, "w:gz") as tar:
        for name in (src_files + notebooks + html_files):
            logging.info(name)
            tar.add(name)

    logging.info("")
    msg = f"Done. Please submit the file {outfile}"
    logging.info("-" * len(msg))
    logging.info(msg)
    logging.info("-" * len(msg))


if __name__ == "__main__":
    create_submit_pkg()
