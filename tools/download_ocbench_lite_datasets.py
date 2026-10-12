"""Download only the 16 selected state-based OCBench lite datasets."""

if __package__:
    from .download_ocbench_datasets import main as download
else:
    from download_ocbench_datasets import main as download


def main(argv=None):
    return download(argv, fixed_suite="lite")


if __name__ == "__main__":
    raise SystemExit(main())
