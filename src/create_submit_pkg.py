import datetime
import glob
import logging
import subprocess
import tarfile

logger = logging.getLogger(__name__)


def create_submit_pkg() -> None:
    # Source files
    src_files = glob.glob("src/*.py")

    # Notebooks
    notebooks = glob.glob("*.ipynb")

    # Genereate HTML files from the notebooks
    for nb in notebooks:
        cmd_line = f"jupyter nbconvert --to html {nb}"

        logger.info(f"executing: {cmd_line}")
        subprocess.check_call(cmd_line, shell=True)

    html_files = glob.glob("*.htm*")

    now = datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h") + "m"
    outfile = f"submission_{now}.tar.gz"
    logger.info(f"Adding files to {outfile}")
    with tarfile.open(outfile, "w:gz") as tar:
        for name in (src_files + notebooks + html_files):
            logger.info(name)
            tar.add(name)

    logger.info("")
    msg = f"Done. Please submit the file {outfile}"
    logger.info("-" * len(msg))
    logger.info(msg)
    logger.info("-" * len(msg))


if __name__ == "__main__":
    create_submit_pkg()
